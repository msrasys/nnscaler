from pathlib import Path
from functools import partial
from subprocess import check_call as _call, check_output
import os
import sys

import shutil
import yaml

call = partial(_call, shell=True)


def train_model(config_dir: Path, save_dir: Path):
    save_dir = save_dir / config_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / 'config.yaml'

    with open(config_file) as f:
        config = yaml.safe_load(f)

    path = Path(config['train']['path']).absolute()
    new_model = path / config['train']['output']
    env = {}
    env.update(os.environ)
    env.update({
        'TRAIN_DATA_DIR': str(Path(os.getenv('TRAIN_DATA_DIR'))),
        'CONFIG_DIR': str(config_dir),
        'SAVE_DIR': str(save_dir),
        'RDZV_ENDPOINT': 'localhost:' + os.getenv('UNUSED_PORT'),
    })
    env.update(config['train'].get('envs', {}))
    for command in config['train']['commands']:
        call(command, env=env, cwd=path)
    shutil.copy2(new_model, save_dir / 'model.pt')


def main(workspace: str, parity_check_dir: str, parity_save_dir: str):
    parity_check_root = Path(parity_check_dir).absolute()
    parity_save_root = Path(parity_save_dir).absolute()
    os.chdir(workspace)
    test_cases = os.getenv('TEST_CASES')
    if test_cases:
        test_cases = test_cases.split(',')
        print(f'Run test cases: {test_cases}')
    else:
        test_cases = None
        print('Run all test cases')
    for d in parity_check_root.glob('*'):
        if not d.is_dir():
            continue
        if not test_cases or d.name in test_cases:
            print(f'Training for {d.name}...')
            train_model(d, parity_save_root)


if __name__ == '__main__':
    if len(sys.argv) !=4:
        print('Usage: python train.py <workspace> <parity_check_dir> <save_dir>')
        exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
