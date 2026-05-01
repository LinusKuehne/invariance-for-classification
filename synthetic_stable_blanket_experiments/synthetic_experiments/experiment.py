from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from .adversary import optimize_attack
from .models import PredictorBundle, build_oracle_predictor, train_predictor
from .plotting import save_plots
from .scm import (
    LinearGaussianStableBlanketSCM,
    NoiseDistribution,
    NonlinearStableBlanketSCM,
)


def configure_torch(num_threads: int = 1) -> None:
    try:
        torch.set_num_threads(num_threads)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


@dataclass
class ExperimentConfig:
    device: str = "cpu"
    torch_num_threads: int = 1
    n_train: int = 4000
    train_sizes: tuple[int, ...] | None = None
    n_val: int = 1000
    n_test: int = 4000
    predictor_hidden_dim: int = 64
    predictor_depth: int = 2
    predictor_lr: float = 1e-3
    predictor_batch_size: int = 256
    predictor_max_epochs: int = 200
    predictor_patience: int = 20
    predictor_weight_decay: float = 1e-5
    attack_hidden_dim: int = 64
    attack_lr: float = 1e-3
    attack_steps: int = 400
    attack_batch_size: int = 512
    attack_restarts: int = 5
    attack_eval_size: int = 10000
    attack_mode: str = "bound"
    intervene_on_x1: bool = True
    include_x6: bool = False
    noise_distribution: NoiseDistribution = "gaussian"
    student_t_df: int = 3
    lineargaussian: bool = False
    simple: bool = False
    bounds: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    costs: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    max_perturbation_bound: float = 2.0
    objectives: tuple[str, ...] = ("signed_error", "mse", "prediction")
    num_runs: int = 3
    output_dir: str = "outputs"


@dataclass
class ResultRow:
    run_id: int
    train_size: int
    method: str
    subset_size: int
    objective: str
    sweep_value: float
    bound: float
    cost: float
    clean_test_mse: float
    attacked_test_mse: float
    attack_objective_value: float
    regularized_attack_value: float
    intervention_strength: float


def _sweep_values(config: ExperimentConfig) -> tuple[str, tuple[float, ...]]:
    if config.attack_mode == "bound":
        return "bound", tuple(float(v) for v in config.bounds)
    if config.attack_mode == "cost":
        return "cost", tuple(float(v) for v in config.costs)
    raise ValueError(f"Unknown attack mode: {config.attack_mode}")


def _train_sizes(config: ExperimentConfig) -> tuple[int, ...]:
    if config.train_sizes is None:
        return (int(config.n_train),)
    return tuple(int(v) for v in config.train_sizes)


def _sample_datasets(
    config: ExperimentConfig,
    scm: NonlinearStableBlanketSCM | LinearGaussianStableBlanketSCM,
    seed: int,
    train_size: int,
):
    train_gen = torch.Generator(device=scm.device)
    val_gen = torch.Generator(device=scm.device)
    test_gen = torch.Generator(device=scm.device)
    train_gen.manual_seed(seed + 1)
    val_gen.manual_seed(seed + 2)
    test_gen.manual_seed(seed + 3)
    x_train, y_train = scm.sample(train_size, generator=train_gen)
    x_val, y_val = scm.sample(config.n_val, generator=val_gen)
    x_test, y_test = scm.sample(config.n_test, generator=test_gen)
    return x_train, y_train, x_val, y_val, x_test, y_test


def _train_all_predictors(
    config: ExperimentConfig,
    scm: NonlinearStableBlanketSCM | LinearGaussianStableBlanketSCM,
    seed: int,
    train_size: int,
) -> list[PredictorBundle]:
    torch.manual_seed(seed)
    sets = {
        "parents": scm.graph_sets.parents,
        "stable_blanket": scm.graph_sets.stable_blanket,
        "all_variables": scm.graph_sets.all_variables,
    }
    predictors: list[PredictorBundle] = []
    if config.lineargaussian:
        test_gen = torch.Generator(device=scm.device)
        test_gen.manual_seed(seed + 3)
        x_test, y_test = scm.sample(config.n_test, generator=test_gen)
        for name, subset in sets.items():
            x_mean, x_std, standardized_weight = scm.conditional_mean_params(subset)
            predictors.append(
                build_oracle_predictor(
                    name=name,
                    subset_indices=subset,
                    standardized_weight=standardized_weight,
                    x_mean=x_mean,
                    x_std=x_std,
                    x_test_full=x_test,
                    y_test=y_test,
                )
            )
        return predictors

    x_train, y_train, x_val, y_val, x_test, y_test = _sample_datasets(
        config, scm, seed, train_size
    )
    for name, subset in sets.items():
        predictors.append(
            train_predictor(
                name=name,
                subset_indices=subset,
                x_train_full=x_train,
                y_train=y_train,
                x_val_full=x_val,
                y_val=y_val,
                x_test_full=x_test,
                y_test=y_test,
                hidden_dim=config.predictor_hidden_dim,
                depth=config.predictor_depth,
                lr=config.predictor_lr,
                batch_size=config.predictor_batch_size,
                max_epochs=config.predictor_max_epochs,
                patience=config.predictor_patience,
                weight_decay=config.predictor_weight_decay,
            )
        )
    return predictors


def run_single_run(
    config: ExperimentConfig, run_id: int, progress: tqdm | None = None
) -> list[ResultRow]:
    configure_torch(config.torch_num_threads)
    if config.lineargaussian:
        scm = LinearGaussianStableBlanketSCM(
            device=config.device,
            intervene_on_x1=config.intervene_on_x1,
            include_x6=config.include_x6,
            noise_distribution=config.noise_distribution,
            student_t_df=config.student_t_df,
        )
    else:
        scm = NonlinearStableBlanketSCM(
            device=config.device,
            intervene_on_x1=config.intervene_on_x1,
            include_x6=config.include_x6,
            noise_distribution=config.noise_distribution,
            student_t_df=config.student_t_df,
        )
    sweep_name, sweep_values = _sweep_values(config)
    rows: list[ResultRow] = []
    for train_size in _train_sizes(config):
        predictors = _train_all_predictors(config, scm, run_id, train_size)
        for predictor in predictors:
            for objective in config.objectives:
                for sweep_value in sweep_values:
                    bound = float(
                        sweep_value
                        if sweep_name == "bound"
                        else config.max_perturbation_bound
                    )
                    cost = float(sweep_value) if sweep_name == "cost" else None
                    attack_result = optimize_attack(
                        scm=scm,
                        predictor=predictor,
                        bound=bound,
                        cost=cost,
                        intervene_on_x1=config.intervene_on_x1,
                        simple=config.simple,
                        objective=objective,  # type: ignore[arg-type]
                        restarts=config.attack_restarts,
                        steps=config.attack_steps,
                        batch_size=config.attack_batch_size,
                        hidden_dim=config.attack_hidden_dim,
                        lr=config.attack_lr,
                        eval_size=config.attack_eval_size,
                        seed=run_id,
                    )
                    rows.append(
                        ResultRow(
                            run_id=run_id,
                            train_size=int(train_size),
                            method=predictor.name,
                            subset_size=len(predictor.subset_indices),
                            objective=objective,
                            sweep_value=float(sweep_value),
                            bound=float(bound),
                            cost=float(cost or 0.0),
                            clean_test_mse=float(predictor.clean_test_mse),
                            attacked_test_mse=float(attack_result.attacked_test_mse),
                            attack_objective_value=float(
                                attack_result.best_objective_value
                            ),
                            regularized_attack_value=float(
                                attack_result.best_regularized_value
                            ),
                            intervention_strength=float(
                                attack_result.intervention_strength
                            ),
                        )
                    )
                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(
                            run=run_id,
                            train_size=int(train_size),
                            method=predictor.name,
                            objective=objective,
                            **{sweep_name: float(sweep_value)},
                        )
    return rows


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["train_size", "method", "objective", "sweep_value"])
        .agg(
            bound=("bound", "mean"),
            cost=("cost", "mean"),
            clean_test_mse_mean=("clean_test_mse", "mean"),
            clean_test_mse_std=("clean_test_mse", "std"),
            attacked_test_mse_mean=("attacked_test_mse", "mean"),
            attacked_test_mse_std=("attacked_test_mse", "std"),
            attack_objective_mean=("attack_objective_value", "mean"),
            attack_objective_std=("attack_objective_value", "std"),
            regularized_attack_mean=("regularized_attack_value", "mean"),
            regularized_attack_std=("regularized_attack_value", "std"),
            intervention_strength_mean=("intervention_strength", "mean"),
            intervention_strength_std=("intervention_strength", "std"),
            n_runs=("run_id", "nunique"),
        )
        .reset_index()
    )
    for prefix in [
        "clean_test_mse",
        "attacked_test_mse",
        "attack_objective",
        "regularized_attack",
        "intervention_strength",
    ]:
        std_col = f"{prefix}_std"
        sem_col = f"{prefix}_sem"
        ci_low_col = f"{prefix}_ci95_low"
        ci_high_col = f"{prefix}_ci95_high"
        mean_col = f"{prefix}_mean"
        summary[std_col] = summary[std_col].fillna(0.0)
        summary[sem_col] = summary[std_col] / summary["n_runs"].clip(lower=1).pow(0.5)
        ci_radius = 1.96 * summary[sem_col]
        summary[ci_low_col] = summary[mean_col] - ci_radius
        summary[ci_high_col] = summary[mean_col] + ci_radius
    return summary.sort_values(["objective", "train_size", "sweep_value", "method"])


def run_experiment_suite(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    configure_torch(config.torch_num_threads)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    run_ids = list(range(config.num_runs))
    _, sweep_values = _sweep_values(config)
    total_tasks = (
        len(run_ids)
        * len(_train_sizes(config))
        * 3
        * len(config.objectives)
        * len(sweep_values)
    )
    progress = tqdm(total=total_tasks, desc="Experiment progress", unit="attack")
    for run_id in run_ids:
        rows.extend(run_single_run(config, run_id, progress=progress))
    progress.close()
    results = pd.DataFrame([asdict(r) for r in rows])
    summary = summarize_results(results)
    results.to_csv(output_dir / "results_per_run.csv", index=False)
    summary.to_csv(output_dir / "results_summary.csv", index=False)
    save_plots(results, output_dir, attack_mode=config.attack_mode)
    pd.DataFrame([asdict(config)]).to_json(
        output_dir / "config.json", orient="records", indent=2
    )
    return results, summary
