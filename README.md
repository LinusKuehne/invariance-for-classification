Code for the NeurIPS submission. The repository has two parts:

- **`scripts/causal_chambers/`** — real-data experiments on the causal chambers dataset (D-spur)
- **`synthetic_stable_blanket_experiments/`** — synthetic SCM experiments; see its own [README](synthetic_stable_blanket_experiments/README.md)

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

Install in editable mode:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

## Reproducing paper figures

### Causal chambers (real-data experiments)

The data files are already included in `scripts/causal_chambers/data/`. Run both scripts from the `scripts/causal_chambers/` directory:

```bash
cd scripts/causal_chambers
```

**Conditional independence test table** (printed as LaTeX to stdout):

```bash
python gcm_ir1_ir2.py
```

This requires R with the `wGCM` package installed.

**Adversarial budget curves figure** (`data/adversarial_budget_curves.{png,pdf}`):

```bash
python adversarial_follower_dspur.py
```

### Synthetic experiments

See [`synthetic_stable_blanket_experiments/README.md`](synthetic_stable_blanket_experiments/README.md) for the full instructions on reproducing those figures.

## Development

```bash
pre-commit install
ruff check .          # lint
ruff check --fix .    # auto-fix
ruff format .         # format
```
