from subprocess import CalledProcessError
import subprocess
import asyncio
import logging
import logging.handlers
from pathlib import Path
import os
import copy
import yaml

logging.basicConfig(
    filemode='a',
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)

logger = logging.getLogger("CubeSystemTest")
result_logger = logging.getLogger("cube_system_test_results")
warning_logger = logging.getLogger("CompareInterface")

# smaller buffer to quick output
buffer_handler = logging.handlers.BufferingHandler(10)
logger.addHandler(buffer_handler)
result_logger.addHandler(buffer_handler)

global_time = None 

class TestUtils:
    @staticmethod
    def execute_command(cmd: str, cwd: str):
        """Execute a command and log the output"""
        try:
            result = subprocess.check_output(cmd, shell=True, cwd=cwd).decode('utf-8').strip()
            return result
        except subprocess.CalledProcessError as e:
            print("An error occurred while trying to execute:", cmd)
            return None

    @staticmethod
    def call(cmds):
        """Call commands async and log the output"""

        if isinstance(cmds, str):
            cmds = [cmds]
        try:
            results = asyncio.run(TestUtils.run_commands_async(cmds))
            for result in results:
                stdout, stderr = result
                if stdout:
                    logger.info(f'{stdout.decode()}')
                if stderr:
                    err_msg = stderr.decode().strip()
                    if ("Traceback (most recent call last):" in err_msg):
                        result_logger.error(f'{err_msg}')
                    else:
                        logger.error(f'{err_msg}')
        except CalledProcessError as e:
            result_logger.error(f"Commands {cmds} failed with error code {e.returncode}")
            raise

    @staticmethod
    async def run_command_async(cmd: str):
        """run a command async and return the output of stdout and stderr"""
        logger.info(f"Running command: {cmd}")
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        return stdout, stderr

    @staticmethod
    async def run_commands_async(cmds: list):
        """run commands async and return the output of stdout and stderr"""
        tasks = [asyncio.ensure_future(TestUtils.run_command_async(cmd)) for cmd in cmds]
        results = await asyncio.gather(*tasks)
        return results

    @staticmethod
    def get_ipv4_address():
        import re
        interface_name = subprocess.check_output("route -n | grep '^0.0.0.0' | awk '{print $8}'", shell=True).decode().strip()
        ifconfig_output = subprocess.check_output(f"ifconfig {interface_name}", shell=True).decode()  
        ip_address_match = re.search(r'inet (\S+)', ifconfig_output)
        ip_pattern = re.compile(
            r'^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
            r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
            r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
            r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        )
        if ip_address_match:
            ip_addr = ip_address_match.group(1)
            if ip_addr and ip_pattern.match(ip_addr):
                return ip_addr

        if os.getenv("CUBE_MASTER_ADDR"):
            ip_addr = os.getenv("CUBE_MASTER_ADDR").strip()
            if ip_addr and ip_pattern.match(ip_addr):
                return ip_addr

        raise RuntimeError(f"cannot get ip address for interface {interface_name}, you can set master_addr manually by setting the environment variable CUBE_MASTER_ADDR")
        
    @staticmethod
    def gen_log_folder(workspace):
        global global_time
        if global_time is None:
            from datetime import datetime
            now = datetime.now()
            global_time = now.strftime("%Y%m%d_%H%M%S")
        log_folder = Path(workspace) / 'cube_test_logs' / global_time
        if not log_folder.exists():
            log_folder.mkdir(parents=True, exist_ok=True)
        return log_folder

    @staticmethod
    def parse_hosts_file(file_path):
        file_path = Path(file_path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r') as file:
            lines = file.readlines()
        ssh_host_list = [line.strip() for line in lines]
        return ssh_host_list

    @staticmethod
    def logger_redirect(logger1, log_folder, filename) -> tuple[str, logging.FileHandler]:
        import logging.handlers
        file_path = f"{log_folder}/{filename}.log"
        result_handler = logging.FileHandler(file_path, 'a')
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        result_handler.setFormatter(formatter)
        logger1.addHandler(result_handler)
        return file_path, result_handler

    @staticmethod
    def merge_dict(dict_a, dict_b):
        a = copy.deepcopy(dict_a)
        b = copy.deepcopy(dict_b)
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    a[key] = TestUtils.merge_dict(a[key], b[key])
                elif b[key] is None or (b[key] == {}):
                    continue
                else:
                    a[key] = b[key]
            else:
                a[key] = b[key]
        return a

    @staticmethod
    def merge_dicts(*dicts):
        result = {}
        for current_dict in dicts:
            result = TestUtils.merge_dict(result, current_dict)
        return result

    @staticmethod
    def load_yaml_file(file_path) -> dict:
        with open(file_path, 'r') as f:
            element = yaml.safe_load(f)
            if isinstance(element, dict):
                TestUtils.recursive_replace_keys(element, '_', '-', 'fairseq')
                TestUtils.recursive_replace_keys(element, '-', '_', 'torchrun')
                TestUtils.recursive_replace_keys(element, '-', '_', 'envs')
                return element
            else:
                raise ValueError(f"Invalid config_file {file_path}")

    @staticmethod
    def recursive_replace_keys(d, old_char, new_char, target_key):
        if target_key in d:
            target_dict = d[target_key]
            d[target_key] = {k.replace(old_char, new_char): v for k, v in target_dict.items()}
        else:
            for _, value in d.items():
                if isinstance(value, dict):
                    TestUtils.recursive_replace_keys(value, old_char, new_char, target_key)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            TestUtils.recursive_replace_keys(item, old_char, new_char, target_key)