# invariance-for-classification

![Build Status](https://github.com/LinusKuehne/invariance-for-classification/actions/workflows/tests.yml/badge.svg)

First, clone the repo:
```bash
git clone https://github.com/LinusKuehne/invariance-for-classification.git
```

Next, create a python virtual environment:
```bash
cd invariance-for-classification
python -m venv venv
source venv/bin/activate
# or venv\Scripts\activate for windows
```

Install in editable mode with `pip`:
```bash
pip install --upgrade pip
pip install -e ".[dev]" # install in editable mode (for modifying the code and seeing the changes immediately) and with developer dependencies
```

Finally, set up pre-commit hooks and verify their installation:
```bash
pre-commit install
pre-commit run --all-files
```

When editing, this can be used as follows:
```bash
ruff check .          # lint
ruff check --fix .    # auto-fix lint issues
ruff format .         # autoformat code
```
