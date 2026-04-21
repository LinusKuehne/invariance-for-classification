from __future__ import annotations

import argparse

from synthetic_experiments.experiment import ExperimentConfig, run_experiment_suite


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic stable blanket adversarial experiments")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=4000)
    p.add_argument("--predictor-hidden-dim", type=int, default=256)
    p.add_argument("--predictor-depth", type=int, default=3)
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
    p.add_argument("--lineargaussian", action="store_true")
    p.add_argument("--simple", action="store_true")
    p.add_argument("--attack-mode", type=str, choices=["bound", "cost"], default="bound")
    p.add_argument("--disable-x1-intervention", action="store_true")
    p.add_argument("--bounds", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--costs", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--max-perturbation-bound", type=float, default=100.0)
    p.add_argument("--objectives", type=str, nargs="+", default=["signed_error", "mse"])
    p.add_argument("--num-runs", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        output_dir=args.output_dir,
        device=args.device,
        torch_num_threads=args.torch_num_threads,
        n_train=args.n_train,
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
