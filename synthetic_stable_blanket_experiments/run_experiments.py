from __future__ import annotations

import argparse

from synthetic_experiments.experiment import ExperimentConfig, run_experiment_suite


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic stable blanket adversarial experiments"
    )
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--n-train", type=int, default=50000)
    p.add_argument(
        "--train-size-sweep",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of training set sizes to sweep; overrides --n-train.",
    )
    p.add_argument("--n-val", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=4000)
    p.add_argument("--predictor-hidden-dim", type=int, default=256)
    p.add_argument("--predictor-depth", type=int, default=5)
    p.add_argument("--predictor-lr", type=float, default=1e-3)
    p.add_argument("--predictor-batch-size", type=int, default=512)
    p.add_argument("--predictor-max-epochs", type=int, default=500)
    p.add_argument("--predictor-patience", type=int, default=20)
    p.add_argument("--predictor-weight-decay", type=float, default=1e-5)
    p.add_argument("--attack-hidden-dim", type=int, default=64)
    p.add_argument("--attack-lr", type=float, default=1e-3)
    p.add_argument("--attack-steps", type=int, default=2000)
    p.add_argument("--attack-batch-size", type=int, default=512)
    p.add_argument("--attack-restarts", type=int, default=5)
    p.add_argument("--attack-eval-size", type=int, default=10000)
    p.add_argument("--X6", action="store_true", dest="include_x6")
    p.add_argument(
        "--noise-distribution",
        type=str,
        choices=["gaussian", "student_t"],
        default="gaussian",
        help="Exogenous noise distribution. student_t adds heavier tails.",
    )
    p.add_argument(
        "--student-t-df",
        type=int,
        default=3,
        help="Degrees of freedom for --noise-distribution student_t; must be > 2.",
    )
    p.add_argument("--lineargaussian", action="store_true")
    p.add_argument("--simple", action="store_true")
    p.add_argument(
        "--x4-uses-x1-x3",
        action="store_true",
        help="Allow the X4 adversary to depend on X1 and X3 in addition to Y, X2, and eps4.",
    )
    p.add_argument(
        "--attack-mode", type=str, choices=["bound", "cost"], default="bound"
    )
    p.add_argument("--disable-x1-intervention", action="store_true")
    p.add_argument("--bounds", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5])
    p.add_argument(
        "--costs", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    p.add_argument("--max-perturbation-bound", type=float, default=100.0)
    p.add_argument(
        "--objectives",
        type=str,
        nargs="+",
        choices=["signed_error", "mse", "prediction"],
        default=["signed_error", "mse", "prediction"],
    )
    p.add_argument("--num-runs", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_sizes = (
        None if args.train_size_sweep is None else tuple(args.train_size_sweep)
    )
    config = ExperimentConfig(
        output_dir=args.output_dir,
        device=args.device,
        torch_num_threads=args.torch_num_threads,
        n_train=args.n_train,
        train_sizes=train_sizes,
        n_val=args.n_val,
        n_test=args.n_test,
        predictor_hidden_dim=args.predictor_hidden_dim,
        predictor_depth=args.predictor_depth,
        predictor_lr=args.predictor_lr,
        predictor_batch_size=args.predictor_batch_size,
        predictor_max_epochs=args.predictor_max_epochs,
        predictor_patience=args.predictor_patience,
        predictor_weight_decay=args.predictor_weight_decay,
        attack_hidden_dim=args.attack_hidden_dim,
        attack_lr=args.attack_lr,
        attack_steps=args.attack_steps,
        attack_batch_size=args.attack_batch_size,
        attack_restarts=args.attack_restarts,
        attack_eval_size=args.attack_eval_size,
        attack_mode=args.attack_mode,
        intervene_on_x1=not args.disable_x1_intervention,
        include_x6=args.include_x6,
        noise_distribution=args.noise_distribution,
        student_t_df=args.student_t_df,
        x4_uses_x1_x3=args.x4_uses_x1_x3,
        lineargaussian=args.lineargaussian,
        simple=args.simple,
        bounds=tuple(args.bounds),
        costs=tuple(args.costs),
        max_perturbation_bound=args.max_perturbation_bound,
        objectives=tuple(args.objectives),
        num_runs=args.num_runs,
    )
    _, summary = run_experiment_suite(config)
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {config.output_dir}")


if __name__ == "__main__":
    main()
