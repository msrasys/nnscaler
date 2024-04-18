sudo echo 'export PATH="$HOME/.local/bin:$PATH"' > $(dirname $(pwd))/.bashrc
source $(dirname $(pwd))/.bashrc
pip install -r requirements-dev.txt
pip install pre-commit
pre-commit install
pre-commit run --all-files
python setup.py develop --user
