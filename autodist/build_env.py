import torch
import os
import shutil
import sys
from datetime import datetime
import subprocess
from pathlib import Path
import torch
from nnscaler.autodist.util import get_node_arch

if bool(int(os.environ.get('PROFILE_COMM', default=0))):
    profile_comm = True
else:
    profile_comm = False


def main():
    base_path = str(Path.home()) + '/.autodist'
    default_path = base_path + '/' + get_node_arch()

    code_path = Path(__file__).parents[1]

    if not os.path.exists(default_path):
        os.makedirs(default_path)
        print('> create folder: ', default_path)
        os.makedirs(default_path + '/plan')
    else:
        print('> folder already exists: ', default_path)

    # profile communication cost
    if profile_comm:
        print('> CUDA device num: ', torch.cuda.device_count())
        for device_num in [2, 4, 8, 16]:
            if device_num > torch.cuda.device_count():
                break
            command = f'torchrun --master_port 21212 --nproc_per_node={device_num} ./comm_profile.py --comm_profile_dir={default_path}/comm'
            output = subprocess.check_output(command, shell=True, text=True)
    else:
        print('> skip communication profiling, using mi200 profile data')
        if os.path.exists(default_path + '/comm'):
            print('> backup existing comm profile data')
            shutil.move(
                default_path + '/comm',
                default_path + f'/comm_back_{str(datetime.now().timestamp())}')
        shutil.copytree(code_path / 'autodist/profile_data/16xmi200/comm', default_path + '/comm')

    print('> build env successfully')


if __name__ == '__main__':
    main()
