"""
Update "nnscaler/version.py" before building the wheel.

Usage 1:

    python update_version.py --nightly

Update version.py to "X.Y.dev{TIMESTAMP}+{GIT_COMMIT}".

Usage 2:

    python update_version.py 1.2
    python update_version.py v1.2b3

Update version.py to the specified version (normalized, leading "v" removed).
It will verify that the release part matches the old version.
"""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess

from packaging.version import Version

project_dir = Path(__file__).parents[2]

def main():
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--nightly', action='store_true')
    group.add_argument('version', nargs='?')
    args = parser.parse_args()

    version_file = Path(project_dir, 'nnscaler/version.py')
    file_content = version_file.read_text()
    version_str = file_content.split('=')[-1].strip()[1:-1]  # "version = 'x'" -> "x"
    repo_version = Version(version_str)

    if args.nightly:
        timestamp = datetime.now().strftime('%y%m%d%H%M')

        r = subprocess.run(
            'git rev-parse --short HEAD'.split(),
            stdout=subprocess.PIPE,
            cwd=project_dir,
            text=True,
        )
        if r.returncode != 0:
            print('[error] failed to get git commit hash')
            exit(1)
        commit = r.stdout.strip()

        new_version_str = f'{repo_version.base_version}.dev{timestamp}+{commit}'

    else:
        arg_version = Version(args.version)

        if repo_version.release != arg_version.release:
            print('[error] version not match')
            print(f'  repo: {version_str} -> {repo_version}')
            print(f'  arg: {args.version} -> {arg_version}')
            exit(1)

        new_version_str = str(arg_version)  # normalize

    file_content = file_content.replace(version_str, new_version_str)
    version_file.write_text(file_content)

if __name__ == '__main__':
    main()
