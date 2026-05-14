# Synthetic experiments for stable blanket vs parents vs all variables

This package implements the nonlinear SCM experiment discussed in the chat.

## SCM

The default graph is:

- `X2 -> X1 -> Y`
- `X2 -> Y`
- `Y -> X3`
- `X2 -> X4 <- Y`
- `X4 -> X5`

Passing `--X6` adds an extra observed variable with parents `X4` and `Y`:

- `X4 -> X6 <- Y`

Predictor sets:

- Parents: `{X1, X2}`
- Stable blanket: `{X1, X2, X3}`
- All variables: `{X1, X2, X3, X4, X5}` by default, and `{X1, X2, X3, X4, X5, X6}` with `--X6`

Directly intervened variables for the adversary:

- `{X1, X4}`

The adversary adds bounded perturbations to the clean mechanisms for `X1` and `X4` using neural networks of the form

`b * tanh(h(parents, noise))`

so the attacked mechanisms become "clean mechanism + bounded deviation". The bound `b` is swept over multiple values, and `b = 0` recovers the clean SCM exactly. `X4` now depends on both `Y` and `X2`, and the adversary can therefore adjust `X4` based on `X2`. `X5` is not directly attacked; it shifts only through the clean downstream mechanism from the attacked `X4`.

Passing `--disable-x1-intervention` turns off the direct intervention on `X1`, so only `X4` is directly attacked.
Passing `--X6` keeps the same intervention set but augments the observed feature vector with the additional descendant `X6`.
By default, the learned `X4` intervention can depend on `(Y, X2, eps4)`.
Passing `--x4-uses-x1-x3` additionally lets it depend on `(X1, X3)`.

There is also a cost-regularized attack mode. In that setting, the adversary optimizes its task objective while paying a penalty proportional to the average squared perturbation size,

`E[delta_X1^2 + delta_X4^2]`,

with regularization weight `c`. The x-axis is then the cost `c` instead of the perturbation bound. A fixed `--max-perturbation-bound` still caps the perturbation amplitudes for numerical stability.

## Adversarial objectives

Three attack objectives are implemented:

1. `signed_error`: minimize `E[Y - f_S(X_S)]`
2. `mse`: maximize `E[(Y - f_S(X_S))^2]`
3. `prediction`: minimize `E[f_S(X_S)]`

Evaluation is always by MSE, and each method is evaluated **only on its own optimized adversary**.

## Files

- `run_experiments.py`: command-line entry point
- `run_minimax_experiment.py`: minimax training on all variables
- `synthetic_experiments/scm.py`: nonlinear SCM and bounded adversarial mechanisms
- `synthetic_experiments/models.py`: predictor training
- `synthetic_experiments/adversary.py`: adversary optimization
- `synthetic_experiments/experiment.py`: orchestration and CSV output
- `synthetic_experiments/plotting.py`: plots of own-adversary MSE vs bound

## Reproducing paper figures

Run the four experiment configurations below (order does not matter; the train-size sweep will take the longest). `run_minimax_experiment.py` is not used for the paper figures.

```bash
python run_experiments.py --output-dir outputs_lingauss --lineargaussian
```

```bash
python run_experiments.py --output-dir outputs_standard
```

```bash
python run_experiments.py --output-dir outputs_train-size-sweep --train-size-sweep 1000 4000 50000
```

```bash
python run_experiments.py --output-dir outputs_x4-uses-x1-x3 --x4-uses-x1-x3
```

Then generate all figures:

```bash
python plot_paper_figures.py
```

Figures are saved to `paper_plots/`.

## Quick smoke test

```bash
python run_experiments.py \
  --output-dir outputs_smoke \
  --torch-num-threads 1 \
  --n-train 600 \
  --n-val 200 \
  --n-test 600 \
  --predictor-max-epochs 40 \
  --predictor-patience 6 \
  --attack-steps 25 \
  --attack-restarts 1 \
  --attack-batch-size 256 \
  --attack-eval-size 1200 \
  --num-runs 1 \
  --disable-x1-intervention \
  --bounds 0.25 0.5 1.0 2.0
```

## Fuller run

```bash
python run_experiments.py \
  --output-dir outputs_full \
  --torch-num-threads 1 \
  --n-train 4000 \
  --n-val 1000 \
  --n-test 4000 \
  --attack-steps 400 \
  --attack-restarts 5 \
  --attack-batch-size 512 \
  --attack-eval-size 10000 \
  --num-runs 3 \
  --bounds 0.25 0.5 1.0 2.0 4.0
```

## Train-size sweep

```bash
python run_experiments.py \
  --output-dir outputs_train_size_sweep \
  --torch-num-threads 1 \
  --train-size-sweep 1000 4000 20000 \
  --n-val 1000 \
  --n-test 4000 \
  --attack-steps 400 \
  --attack-restarts 5 \
  --attack-batch-size 512 \
  --attack-eval-size 10000 \
  --num-runs 3 \
  --bounds 0.25 0.5 1.0 2.0
```

This repeats the leader training and follower attack for each requested training
set size. The plots place all method-by-train-size curves for a given objective
in one figure, and the CSV outputs include a `train_size` column for custom
plotting later.

To give the clean training distribution heavier tails, use Student-t exogenous
noise. The samples are rescaled to have variance one, matching the Gaussian
default:

```bash
python run_experiments.py \
  --output-dir outputs_train_size_sweep_student_t \
  --noise-distribution student_t \
  --student-t-df 3 \
  --train-size-sweep 1000 4000 20000 \
  --n-val 1000 \
  --n-test 4000 \
  --attack-steps 400 \
  --attack-restarts 5 \
  --num-runs 3 \
  --bounds 0.25 0.5 1.0 2.0
```

## Cost-regularized run

```bash
python run_experiments.py \
  --output-dir outputs_cost \
  --attack-mode cost \
  --torch-num-threads 1 \
  --n-train 4000 \
  --n-val 1000 \
  --n-test 4000 \
  --attack-steps 400 \
  --attack-restarts 5 \
  --attack-batch-size 512 \
  --attack-eval-size 10000 \
  --num-runs 3 \
  --max-perturbation-bound 2.0 \
  --costs 0.0 0.01 0.05 0.1 0.25 0.5 1.0
```

## Minimax-trained full-model run

```bash
python run_minimax_experiment.py \
  --output-dir outputs_minimax \
  --attack-mode bound \
  --torch-num-threads 1 \
  --n-train 4000 \
  --n-val 1000 \
  --n-test 4000 \
  --minimax-steps 2000 \
  --adversary-reinit-interval 0 \
  --train-intervention-bound 1.0 \
  --methods parents stable_blanket all_variables minimax_all_variables \
  --attack-steps 400 \
  --attack-restarts 5 \
  --num-runs 3 \
  --bounds 0.25 0.5 1.0 2.0
```

This trains a single predictor on the full variable set using alternating minimax updates against an intervention-constrained adversary, then evaluates it under a fresh best-response MSE attack.
By default, the minimax model is trained separately for each sweep value (`bound` or `cost`). In bound mode, passing `--train-intervention-bound <b_train>` instead trains one minimax model per run at the fixed training bound `b_train` and then evaluates that same model across all requested evaluation bounds.
Use `--methods` to choose which methods to compare. In addition to `minimax_all_variables`, there is also `minimax_learned_scm_all_variables`, which first fits the structural equations from the clean training data using the known graph and then runs minimax training against that learned surrogate SCM.

## Output files

- `results_per_run.csv`
- `results_summary.csv`
- `mse_vs_bound_signed_error.png`
- `mse_vs_bound_mse.png`
- `mse_vs_cost_signed_error.png` (cost mode)
- `mse_vs_cost_mse.png` (cost mode)
- `config.json`

The plots show mean attacked MSE with 95% confidence intervals across runs.

## Expected qualitative behavior

- On the clean SCM, `all_variables` should usually achieve the lowest test MSE.
- As the intervention bound increases, `all_variables` should worsen under its own adversary.
- `stable_blanket` is typically flatter across bounds because it ignores the mutable descendants while still using the stable child `X3`.
- `parents` is also flat across bounds, but can be worse than `stable_blanket` because it discards `X3`.
