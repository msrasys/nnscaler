import os
from pathlib import Path
import sys

import torch


def parity_check(task_name, ground_truth_model_file, new_model_file):
    gt_ckpt = torch.load(ground_truth_model_file, map_location='cpu', weights_only=False)
    new_ckpt = torch.load(new_model_file, map_location='cpu', weights_only=False)
    if 'model' in gt_ckpt:
        gt_model = gt_ckpt['model']
        new_model = new_ckpt['model']
    elif 'state_dict' in gt_ckpt:
        gt_model = gt_ckpt['state_dict']
        new_model = new_ckpt['state_dict']
    for name in gt_model:
        if not torch.allclose(gt_model[name], new_model[name], rtol=1e-06, atol=1e-06):
            raise Exception(f'{task_name} failed: {name} mismatch (rtol=1e-06, atol=1e-06)')
    print('All weights match (rtol=1e-06, atol=1e-06)')


def main(gt_dir: str, new_dir: str):
    new_dir = Path(new_dir).absolute()

    test_cases = os.getenv('TEST_CASES')
    if test_cases:
        test_cases = test_cases.split(',')
        print(f'Check test cases: {test_cases}')
    else:
        test_cases = None
        print('Check all test cases')
    passed = []
    for d in Path(gt_dir).glob('*'):
        if not d.is_dir():
            continue
        if not test_cases or d.name in test_cases:
            print(f'Checking for {d.name}...')
            parity_check(d.name, d / 'model.pt', new_dir / d.name / 'model.pt')
            passed.append(d.name)
    print(f'All passed: {passed}')


if __name__ == '__main__':
    if len(sys.argv) !=3:
        print('Usage: python check.py <gt_dir> <new_dir>')
        exit(1)
    main(sys.argv[1], sys.argv[2])
