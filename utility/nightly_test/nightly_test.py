from test_utils import TestUtils
from azure.communication.email import EmailClient
from subprocess import CalledProcessError
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import base64
import zipfile
import json
import os
import sys

sender_address = "DoNotReply@ca1e34f6-1a6d-4181-8b16-692dbe193525.azurecomm.net"

def zip_folder(folder_path, output_path):
    """ Zip the folder to the output path
        Args:
            folder_path: the folder path
            output_path: the output path
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), os.path.dirname(folder_path))
                zipf.write(os.path.join(root, file), arcname=relative_path)

def get_branch_commit(repo_path, branch_name = None, days_ago = 0):
    """ Get the branch name or commit ID of the branch_name that is days_ago
        Args:
            repo_path: the path of the git repo
            branch_name: the branch name, if not provided return the branch name
            days_ago: the days ago, 0 means get current commit ID of the branch
        Returns:
            The branch name of the commit ID
    """
    if branch_name is None:
        git_command = 'git rev-parse --abbrev-ref HEAD'
        return TestUtils.execute_command(git_command, repo_path)
    elif days_ago == 0:
        git_command = 'git rev-parse HEAD'
        return TestUtils.execute_command(git_command, repo_path)
    else:
        before_date = (datetime.now() - timedelta(days=int(days_ago)) + timedelta(hours=15)).strftime('%Y-%m-%d %H:%M:%S')    # add 15 hours to align with Beijing time
        git_command = 'git fetch && git rev-list -n 1 --before="{}" {}'.format(before_date, branch_name)
        return TestUtils.execute_command(git_command, repo_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Running Nightly Test')
    parser.add_argument('-w', '--workspace', required=True, help='workspace for nightly test')
    parser.add_argument('-d', '--data-path', required=True, help='dataset path')

    parser.add_argument('-n', '--nnscaler-commit-id', help='nnscaler commit id, decide the version of nnscaler for unit test and example parity-check')

    parser.add_argument('-u', '--unit-test', default=False, action=argparse.BooleanOptionalAction, help='unit test for nnscaler')

    parser.add_argument('-ep', '--example-parity-check', default=False, action=argparse.BooleanOptionalAction, help='example parity check for nnscaler. It will compare nnscaler <nnscaler-commit-id> or main with <cube-branch-gt> or main')
    
    # Keeping old argument name for compatibility if needed, but help text updated
    parser.add_argument('-p2', '--parity-check2', dest='example_parity_check', action='store_true', help='Alias for --example-parity-check')

    parser.add_argument('-pb', '--parity-check-conda-base', help='base conda environment for parity check, needed if example-parity-check is True')
    parser.add_argument('-ngt', '--cube-branch-gt', default='main', help='cube branch for ground truth, default is main')

    parser.add_argument('-e', '--email-connect-string', help='email connect string for sending email address')
    parser.add_argument('-et', '--email-to', action='append', default=[], help='multiple -et will be combined')
    parser.add_argument('-ec', '--email-cc', action='append', default=[], help='multiple -ec will be combined')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    workspace =  Path(args.workspace).expanduser()
    data_path = Path(args.data_path).expanduser()
    if not workspace.exists():
        raise ValueError(f"Invalid workspace path: {workspace}")
    if not data_path.exists():
        raise ValueError(f"Invalid data_path path: {data_path}")
    log_folder = TestUtils.gen_log_folder(workspace)

    # Assuming nnscaler is cloned as "nnscaler" in the workspace
    nnscaler_repo_path = workspace / "nnscaler"
    pytest_dir = nnscaler_repo_path / "tests"
    
    script_dir = Path(__file__).parent.absolute()
    parity_alert_script = script_dir / "parity_alert_examples/parity_alert.sh"
    parity_check_cases_dir = script_dir / "parity_alert_examples/test_cases"

    if not pytest_dir.exists():
        raise ValueError(f"Invalid pytest_dir path: {pytest_dir}")
    if not parity_alert_script.exists():
        raise ValueError(f"Invalid parity_alert_script path: {parity_alert_script}")

    if args.nnscaler_commit_id:
        cmd = f"parallel-ssh -x -q -h ~/.pssh_hosts_files git -C {nnscaler_repo_path} checkout {args.nnscaler_commit_id}"
        TestUtils.call([cmd])

    with open(TestUtils.gen_log_folder(workspace) / "nightly_test.log", 'a') as nightly_test_file:
        nnscaler_branch = get_branch_commit(nnscaler_repo_path)
        nnscaler_commit_id = get_branch_commit(nnscaler_repo_path, nnscaler_branch)
        nightly_test_file.write(f"nnscaler on branch {nnscaler_branch}, commit ID {nnscaler_commit_id}\n\n")
        nightly_test_file.flush()

        # Run Unit Test
        pytest_output = ""
        if args.unit_test:
            pytest_cmd = f"{sys.executable} -m pytest {pytest_dir}"
            try:
                pytest_log_file = log_folder / "pytest.log"
                with open(pytest_log_file, 'w') as f:
                    # Run pytest from inside nnscaler repo
                    result = subprocess.run([sys.executable, '-m', 'pytest', '-v', str(pytest_dir)], stdout=f, stderr=f, cwd=nnscaler_repo_path)
                if result.returncode != 0:
                    pytest_output = f"NNScaler Unit test didn't pass, see {pytest_log_file.name} for more details."
                else:
                    pytest_output = "NNScaler Unit test passed"
            except CalledProcessError as e:
                pytest_output = f"Command {pytest_cmd} failed with error code {e.returncode}"
            finally:
                nightly_test_file.write(pytest_output + "\n")

        # Run Example Parity Check
        parity_alert_output = ""
        if args.example_parity_check:
            tmp_parity_check = workspace / 'tmp_example_parity_check'
            if os.path.isdir(tmp_parity_check):
                import shutil
                shutil.rmtree(tmp_parity_check)
            
            if not args.nnscaler_commit_id:
                # If not specified, get the current one for consistency in logging/checking
                 args.nnscaler_commit_id = get_branch_commit(nnscaler_repo_path, "origin/main", 0)

            nightly_test_file.write(f"Example Parity check:\nnnscaler commit ID: {args.nnscaler_commit_id}" + "\n")

            parity_check_cmd = f"bash {parity_alert_script} {tmp_parity_check} {data_path} {parity_check_cases_dir} --cube-branch {args.nnscaler_commit_id} --cube-branch-gt {args.cube_branch_gt} --conda-base {args.parity_check_conda_base}"
            
            env = os.environ.copy()
            # Assuming we might need to set PYTHONPATH if needed for some scripts, but usually the parity script handles env setup
            # But let's keep consistency if we copied parity_alert_examples which relies on some imports
            try:
                parity_log_file = log_folder / "example_parity_check.log"
                with open(parity_log_file, 'w') as f:
                    # CWD to the directory of parity_alert_examples for any relative path assumptions inside train.py potentially
                    cwd_path = script_dir / "parity_alert_examples"
                    result = subprocess.run(parity_check_cmd, stdout=f, stderr=f, shell=True, env=env, cwd=cwd_path)
                if result.returncode != 0:
                    parity_alert_output = f"Example Parity Check didn't pass, see {parity_log_file.name} for more details."
                else:
                    parity_alert_output = "Example Parity Check passed"
            except CalledProcessError as e:
                parity_alert_output = f"Command {parity_check_cmd} failed with error code {e.returncode}"
            finally:
                nightly_test_file.write(parity_alert_output + "\n")

        nightly_test_file.flush()

        # Send email
        if args.email_connect_string:
            if not args.email_to:
                raise ValueError(f"Invalid email_to: {args.email_to}")
            zip_output = log_folder.parent / 'nightly_test_logs.zip'
            zip_folder(log_folder, zip_output)
            with open(zip_output, "rb") as file:
                zip_b64encoded = base64.b64encode(file.read())

            html_output = """
            <html>
            <head>
                <title>Test Results</title>
                <style>
                    body { font-family: Arial, sans-serif; }
                    .failed { color: red; }
                    .passed { }
                    ul { list-style-type: none; }
                </style>
            </head>
            <body>
            """

            if args.unit_test:
                pytest_html_message = f"""<h3>NNScaler Unit Test</h3><p{' class="failed"' if pytest_output != "NNScaler Unit test passed" else ""}>{pytest_output}</p>"""
                html_output += pytest_html_message
            
            if args.example_parity_check:
                parity_html_message = f"""<h3>Example Parity Check</h3><p{' class="failed"' if parity_alert_output != "Example Parity Check passed" else ""}>{parity_alert_output}</p>"""
                html_output += parity_html_message

            html_output +="""</body></html>"""

            message = {
                "senderAddress": sender_address,
                "recipients":  {
                    "to": [{ "address": t } for t in args.email_to],
                    "cc": [{ "address": t } for t in args.email_cc]
                },
                "content": {
                    "subject": "Nightly Test Notification",
                    "html": html_output
                },
                "attachments": [
                    {
                        "name": "attachment.zip",
                        "contentType": "application/zip",
                        "contentInBase64": zip_b64encoded.decode()
                    }
                ]
            }

            try:
                POLLER_WAIT_TIME = 10
                client = EmailClient.from_connection_string(args.email_connect_string)
                poller = client.begin_send(message)
                time_elapsed = 0
                while not poller.done():
                    poller.wait(POLLER_WAIT_TIME)
                    time_elapsed += POLLER_WAIT_TIME
                    if time_elapsed > 18 * POLLER_WAIT_TIME:
                        raise RuntimeError("Polling timed out.")
                if poller.result()["status"] == "Succeeded":
                    nightly_test_file.write(f"Successfully sent the email (operation id: {poller.result()['id']})")
                else:
                    raise RuntimeError(str(poller.result()["error"]))
            except Exception as ex:
                nightly_test_file.write(str(ex))
        else:
            nightly_test_file.write("No email connection string provided, skip sending email")
