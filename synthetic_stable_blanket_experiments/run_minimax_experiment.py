from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from synthetic_experiments.adversary import optimize_attack
from synthetic_experiments.experiment import configure_torch
from synthetic_experiments.models import MLPRegressor, PredictorBundle, train_predictor
from synthetic_experiments.scm import AttackMechanisms, NonlinearStableBlanketSCM

METHOD_ORDER = [
    "parents",
    "stable_blanket",
    "all_variables",
    "minimax_all_variables",
    "minimax_learned_scm_all_variables",
]


@dataclass
class MinimaxConfig:
    output_dir: str = "outputs_minimax"
    device: str = "cpu"
    torch_num_threads: int = 1
    n_train: int = 20000
    n_val: int = 2000
    n_test: int = 4000
    predictor_hidden_dim: int = 128
    predictor_depth: int = 3
    predictor_lr: float = 1e-3
    predictor_weight_decay: float = 1e-5
    attack_hidden_dim: int = 64
    attack_lr: float = 1e-3
    attack_mode: str = "bound"
    intervene_on_x1: bool = True
    include_x6: bool = False
    train_intervention_bound: float | None = None
    bounds: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    costs: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    max_perturbation_bound: float = 2.0
    minimax_steps: int = 2000
    predictor_steps_per_iter: int = 1
    adversary_steps_per_iter: int = 1
    adversary_reinit_interval: int = 0
    minimax_batch_size: int = 512
    minimax_validation_interval: int = 50
    val_attack_restarts: int = 2
    val_attack_steps: int = 100
    val_attack_batch_size: int = 256
    attack_restarts: int = 5
    attack_steps: int = 400
    attack_batch_size: int = 512
    attack_eval_size: int = 10000
    num_runs: int = 3
    methods: tuple[str, ...] = tuple(METHOD_ORDER)


@dataclass
class MinimaxResultRow:
    run_id: int
    method: str
    sweep_value: float
    bound: float
    cost: float
    clean_test_mse: float
    attacked_test_mse: float
    attack_objective_value: float
    regularized_attack_value: float
    intervention_strength: float


@dataclass
class StandardizedEquationModel:
    model: MLPRegressor
    x_mean: torch.Tensor
    x_std: torch.Tensor

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.x_mean) / self.x_std)


class LearnedMinimaxSCM:
    def __init__(
        self,
        *,
        device: str,
        intervene_on_x1: bool,
        include_x6: bool,
        graph_sets,
        x2_pool: torch.Tensor,
        x1_model: StandardizedEquationModel,
        x1_residual_pool: torch.Tensor,
        y_model: StandardizedEquationModel,
        y_residual_pool: torch.Tensor,
        x3_model: StandardizedEquationModel,
        x3_residual_pool: torch.Tensor,
        x4_model: StandardizedEquationModel,
        x4_residual_pool: torch.Tensor,
        x5_model: StandardizedEquationModel,
        x5_residual_pool: torch.Tensor,
        x6_model: StandardizedEquationModel | None,
        x6_residual_pool: torch.Tensor | None,
    ) -> None:
        self.device = torch.device(device)
        self.intervene_on_x1 = intervene_on_x1
        self.include_x6 = include_x6
        self.graph_sets = graph_sets
        self.x2_pool = x2_pool.to(self.device)
        self.x1_model = x1_model
        self.x1_residual_pool = x1_residual_pool.to(self.device)
        self.y_model = y_model
        self.y_residual_pool = y_residual_pool.to(self.device)
        self.x3_model = x3_model
        self.x3_residual_pool = x3_residual_pool.to(self.device)
        self.x4_model = x4_model
        self.x4_residual_pool = x4_residual_pool.to(self.device)
        self.x5_model = x5_model
        self.x5_residual_pool = x5_residual_pool.to(self.device)
        self.x6_model = x6_model
        self.x6_residual_pool = None if x6_residual_pool is None else x6_residual_pool.to(self.device)
        self.x1_noise_scale = self.x1_residual_pool.std().clamp_min(1e-6)
        self.x4_noise_scale = self.x4_residual_pool.std().clamp_min(1e-6)

    def _sample_from_pool(
        self,
        pool: torch.Tensor,
        n: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        indices = torch.randint(0, pool.shape[0], (n,), generator=generator, device=self.device)
        return pool.index_select(0, indices)

    def sample(
        self,
        n: int,
        attack: AttackMechanisms | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, _, _ = self.sample_with_intervention_info(n, attack=attack, generator=generator)
        return x, y

    def sample_with_intervention_info(
        self,
        n: int,
        attack: AttackMechanisms | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x2 = self._sample_from_pool(self.x2_pool, n, generator)
        eps1 = self._sample_from_pool(self.x1_residual_pool, n, generator)
        clean_x1 = self.x1_model.predict(x2) + eps1
        noise_x1 = eps1 / self.x1_noise_scale
        delta_x1 = torch.zeros_like(clean_x1)
        if attack is not None and self.intervene_on_x1:
            delta_x1 = attack.perturb_x1(x2, noise_x1)
        x1 = clean_x1 + delta_x1

        eps_y = self._sample_from_pool(self.y_residual_pool, n, generator)
        y = self.y_model.predict(torch.cat([x1, x2], dim=1)) + eps_y

        eps3 = self._sample_from_pool(self.x3_residual_pool, n, generator)
        x3 = self.x3_model.predict(y) + eps3

        eps4 = self._sample_from_pool(self.x4_residual_pool, n, generator)
        clean_x4 = self.x4_model.predict(torch.cat([y, x2], dim=1)) + eps4
        noise_x4 = eps4 / self.x4_noise_scale
        delta_x4 = torch.zeros_like(clean_x4)
        if attack is not None:
            delta_x4 = attack.perturb_x4(y, x2, noise_x4)
        x4 = clean_x4 + delta_x4

        eps5 = self._sample_from_pool(self.x5_residual_pool, n, generator)
        x5 = self.x5_model.predict(x4) + eps5

        pieces = [x1, x2, x3, x4, x5]
        if self.include_x6:
            assert self.x6_model is not None and self.x6_residual_pool is not None
            eps6 = self._sample_from_pool(self.x6_residual_pool, n, generator)
            x6 = self.x6_model.predict(torch.cat([x4, y], dim=1)) + eps6
            pieces.append(x6)

        x = torch.cat(pieces, dim=1)
        return x, y, delta_x1, delta_x4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimax adversarial training on all variables")
    p.add_argument("--output-dir", type=str, default="outputs_minimax")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=4000)
    p.add_argument("--predictor-hidden-dim", type=int, default=64)
    p.add_argument("--predictor-depth", type=int, default=2)
    p.add_argument("--predictor-lr", type=float, default=1e-3)
    p.add_argument("--predictor-weight-decay", type=float, default=1e-5)
    p.add_argument("--attack-hidden-dim", type=int, default=64)
    p.add_argument("--attack-lr", type=float, default=1e-3)
    p.add_argument("--attack-mode", type=str, choices=["bound", "cost"], default="bound")
    p.add_argument("--disable-x1-intervention", action="store_true")
    p.add_argument("--X6", action="store_true", dest="include_x6")
    p.add_argument("--train-intervention-bound", type=float, default=None)
    p.add_argument("--bounds", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--costs", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--max-perturbation-bound", type=float, default=2.0)
    p.add_argument("--minimax-steps", type=int, default=2000)
    p.add_argument("--predictor-steps-per-iter", type=int, default=1)
    p.add_argument("--adversary-steps-per-iter", type=int, default=1)
    p.add_argument("--adversary-reinit-interval", type=int, default=0)
    p.add_argument("--minimax-batch-size", type=int, default=512)
    p.add_argument("--minimax-validation-interval", type=int, default=50)
    p.add_argument("--val-attack-restarts", type=int, default=2)
    p.add_argument("--val-attack-steps", type=int, default=100)
    p.add_argument("--val-attack-batch-size", type=int, default=256)
    p.add_argument("--attack-restarts", type=int, default=5)
    p.add_argument("--attack-steps", type=int, default=400)
    p.add_argument("--attack-batch-size", type=int, default=512)
    p.add_argument("--attack-eval-size", type=int, default=10000)
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--methods", type=str, nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER)
    return p.parse_args()


def _sweep_values(config: MinimaxConfig) -> tuple[str, tuple[float, ...]]:
    if config.attack_mode == "bound":
        return "bound", tuple(float(v) for v in config.bounds)
    if config.attack_mode == "cost":
        return "cost", tuple(float(v) for v in config.costs)
    raise ValueError(f"Unknown attack mode: {config.attack_mode}")


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def _mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((y - pred) ** 2)


def _strength(delta_x1: torch.Tensor, delta_x4: torch.Tensor) -> torch.Tensor:
    return torch.mean(delta_x1.square() + delta_x4.square())


def _standardize(x: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
    return (x - x_mean) / x_std


def _fit_equation_model(
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    hidden_dim: int,
    depth: int,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    weight_decay: float,
) -> StandardizedEquationModel:
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train_std = (x_train - x_mean) / x_std
    x_val_std = (x_val - x_mean) / x_std

    model = MLPRegressor(x_train.shape[1], hidden_dim=hidden_dim, depth=depth).to(x_train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_train_std, y_train), batch_size=batch_size, shuffle=True, num_workers=0)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience_count = 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val_std), y_val).item()

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return StandardizedEquationModel(model=model, x_mean=x_mean, x_std=x_std)


def _sample_clean_datasets(config: MinimaxConfig, scm: NonlinearStableBlanketSCM, seed: int) -> tuple[torch.Tensor, ...]:
    train_gen = torch.Generator(device=scm.device)
    val_gen = torch.Generator(device=scm.device)
    test_gen = torch.Generator(device=scm.device)
    train_gen.manual_seed(seed + 1)
    val_gen.manual_seed(seed + 2)
    test_gen.manual_seed(seed + 3)
    x_train, y_train = scm.sample(config.n_train, generator=train_gen)
    x_val, y_val = scm.sample(config.n_val, generator=val_gen)
    x_test, y_test = scm.sample(config.n_test, generator=test_gen)
    return x_train, y_train, x_val, y_val, x_test, y_test


def _estimate_training_scm(
    config: MinimaxConfig,
    *,
    true_scm: NonlinearStableBlanketSCM,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> LearnedMinimaxSCM:
    fit_kwargs = dict(
        hidden_dim=config.predictor_hidden_dim,
        depth=config.predictor_depth,
        lr=config.predictor_lr,
        batch_size=config.minimax_batch_size,
        max_epochs=max(50, min(500, config.minimax_steps)),
        patience=20,
        weight_decay=config.predictor_weight_decay,
    )

    x1_model = _fit_equation_model(
        x_train=x_train[:, [1]],
        y_train=x_train[:, [0]],
        x_val=x_val[:, [1]],
        y_val=x_val[:, [0]],
        **fit_kwargs,
    )
    y_model = _fit_equation_model(
        x_train=torch.cat([x_train[:, [0]], x_train[:, [1]]], dim=1),
        y_train=y_train,
        x_val=torch.cat([x_val[:, [0]], x_val[:, [1]]], dim=1),
        y_val=y_val,
        **fit_kwargs,
    )
    x3_model = _fit_equation_model(
        x_train=y_train,
        y_train=x_train[:, [2]],
        x_val=y_val,
        y_val=x_val[:, [2]],
        **fit_kwargs,
    )
    x4_model = _fit_equation_model(
        x_train=torch.cat([y_train, x_train[:, [1]]], dim=1),
        y_train=x_train[:, [3]],
        x_val=torch.cat([y_val, x_val[:, [1]]], dim=1),
        y_val=x_val[:, [3]],
        **fit_kwargs,
    )
    x5_model = _fit_equation_model(
        x_train=x_train[:, [3]],
        y_train=x_train[:, [4]],
        x_val=x_val[:, [3]],
        y_val=x_val[:, [4]],
        **fit_kwargs,
    )

    x6_model: StandardizedEquationModel | None = None
    x6_residual_pool: torch.Tensor | None = None
    if config.include_x6:
        x6_model = _fit_equation_model(
            x_train=torch.cat([x_train[:, [3]], y_train], dim=1),
            y_train=x_train[:, [5]],
            x_val=torch.cat([x_val[:, [3]], y_val], dim=1),
            y_val=x_val[:, [5]],
            **fit_kwargs,
        )
        with torch.no_grad():
            x6_residual_pool = x_train[:, [5]] - x6_model.predict(torch.cat([x_train[:, [3]], y_train], dim=1))

    with torch.no_grad():
        x1_residual_pool = x_train[:, [0]] - x1_model.predict(x_train[:, [1]])
        y_residual_pool = y_train - y_model.predict(torch.cat([x_train[:, [0]], x_train[:, [1]]], dim=1))
        x3_residual_pool = x_train[:, [2]] - x3_model.predict(y_train)
        x4_residual_pool = x_train[:, [3]] - x4_model.predict(torch.cat([y_train, x_train[:, [1]]], dim=1))
        x5_residual_pool = x_train[:, [4]] - x5_model.predict(x_train[:, [3]])

    return LearnedMinimaxSCM(
        device=config.device,
        intervene_on_x1=config.intervene_on_x1,
        include_x6=config.include_x6,
        graph_sets=true_scm.graph_sets,
        x2_pool=x_train[:, [1]].detach().clone(),
        x1_model=x1_model,
        x1_residual_pool=x1_residual_pool.detach().clone(),
        y_model=y_model,
        y_residual_pool=y_residual_pool.detach().clone(),
        x3_model=x3_model,
        x3_residual_pool=x3_residual_pool.detach().clone(),
        x4_model=x4_model,
        x4_residual_pool=x4_residual_pool.detach().clone(),
        x5_model=x5_model,
        x5_residual_pool=x5_residual_pool.detach().clone(),
        x6_model=x6_model,
        x6_residual_pool=None if x6_residual_pool is None else x6_residual_pool.detach().clone(),
    )


def train_minimax_predictor(
    config: MinimaxConfig,
    *,
    run_id: int,
    bound: float,
    cost: float | None,
) -> PredictorBundle:
    return _train_minimax_predictor(
        config,
        run_id=run_id,
        bound=bound,
        cost=cost,
        method_name="minimax_all_variables",
        use_learned_scm=False,
    )


def train_learned_scm_minimax_predictor(
    config: MinimaxConfig,
    *,
    run_id: int,
    bound: float,
    cost: float | None,
) -> PredictorBundle:
    return _train_minimax_predictor(
        config,
        run_id=run_id,
        bound=bound,
        cost=cost,
        method_name="minimax_learned_scm_all_variables",
        use_learned_scm=True,
    )


def _train_minimax_predictor(
    config: MinimaxConfig,
    *,
    run_id: int,
    bound: float,
    cost: float | None,
    method_name: str,
    use_learned_scm: bool,
) -> PredictorBundle:
    configure_torch(config.torch_num_threads)
    torch.manual_seed(run_id)
    true_scm = NonlinearStableBlanketSCM(
        device=config.device,
        intervene_on_x1=config.intervene_on_x1,
        include_x6=config.include_x6,
    )
    x_train, y_train, x_val, y_val, x_test, y_test = _sample_clean_datasets(config, true_scm, run_id)
    training_scm = (
        _estimate_training_scm(config, true_scm=true_scm, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        if use_learned_scm
        else true_scm
    )

    subset_indices = true_scm.graph_sets.all_variables
    x_train_full = x_train[:, subset_indices]
    x_mean = x_train_full.mean(dim=0, keepdim=True)
    x_std = x_train_full.std(dim=0, keepdim=True).clamp_min(1e-6)

    predictor = MLPRegressor(
        len(subset_indices),
        hidden_dim=config.predictor_hidden_dim,
        depth=config.predictor_depth,
    ).to(true_scm.device)
    def build_attack(seed_offset: int) -> AttackMechanisms:
        torch.manual_seed(run_id + seed_offset)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_id + seed_offset)
        return AttackMechanisms(
            bound=bound,
            hidden_dim=config.attack_hidden_dim,
            intervene_on_x1=config.intervene_on_x1,
        ).to(true_scm.device)

    attack = build_attack(seed_offset=22222)

    predictor_optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=config.predictor_lr,
        weight_decay=config.predictor_weight_decay,
    )
    attack_optimizer = torch.optim.Adam(attack.parameters(), lr=config.attack_lr)

    train_gen = torch.Generator(device=true_scm.device)
    train_gen.manual_seed(run_id + 12345)

    best_predictor_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")

    for step_idx in range(config.minimax_steps):
        if config.adversary_reinit_interval > 0 and step_idx > 0 and step_idx % config.adversary_reinit_interval == 0:
            attack = build_attack(seed_offset=22222 + step_idx)
            attack_optimizer = torch.optim.Adam(attack.parameters(), lr=config.attack_lr)

        _set_requires_grad(attack, False)
        _set_requires_grad(predictor, True)
        for _ in range(config.predictor_steps_per_iter):
            predictor_optimizer.zero_grad(set_to_none=True)
            x_adv, y_adv, _, _ = training_scm.sample_with_intervention_info(
                config.minimax_batch_size,
                attack=attack,
                generator=train_gen,
            )
            pred = predictor(_standardize(x_adv[:, subset_indices], x_mean, x_std))
            loss = _mse(pred, y_adv)
            loss.backward()
            predictor_optimizer.step()

        _set_requires_grad(predictor, False)
        _set_requires_grad(attack, True)
        for _ in range(config.adversary_steps_per_iter):
            attack_optimizer.zero_grad(set_to_none=True)
            x_adv, y_adv, delta_x1, delta_x4 = training_scm.sample_with_intervention_info(
                config.minimax_batch_size,
                attack=attack,
                generator=train_gen,
            )
            pred = predictor(_standardize(x_adv[:, subset_indices], x_mean, x_std))
            objective = _mse(pred, y_adv)
            attack_loss = -objective
            if cost is not None:
                attack_loss = attack_loss + cost * _strength(delta_x1, delta_x4)
            attack_loss.backward()
            attack_optimizer.step()

        should_validate = ((step_idx + 1) % max(1, config.minimax_validation_interval) == 0) or (
            step_idx + 1 == config.minimax_steps
        )
        if should_validate:
            predictor.eval()
            robust_bundle = PredictorBundle(
                name=method_name,
                subset_indices=tuple(int(i) for i in subset_indices),
                model=predictor,
                x_mean=x_mean,
                x_std=x_std,
                clean_test_mse=0.0,
            )
            robust_val = optimize_attack(
                scm=training_scm,
                predictor=robust_bundle,
                bound=bound,
                cost=cost,
                intervene_on_x1=config.intervene_on_x1,
                objective="mse",
                restarts=config.val_attack_restarts,
                steps=config.val_attack_steps,
                batch_size=config.val_attack_batch_size,
                hidden_dim=config.attack_hidden_dim,
                lr=config.attack_lr,
                eval_size=config.n_val,
                seed=run_id + 777 + step_idx,
            )
            if robust_val.attacked_test_mse < best_val:
                best_val = robust_val.attacked_test_mse
                best_predictor_state = {k: v.detach().clone() for k, v in predictor.state_dict().items()}

    assert best_predictor_state is not None
    predictor.load_state_dict(best_predictor_state)
    predictor.eval()

    with torch.no_grad():
        clean_test_mse = _mse(predictor(_standardize(x_test[:, subset_indices], x_mean, x_std)), y_test).item()

    bundle = PredictorBundle(
        name=method_name,
        subset_indices=tuple(int(i) for i in subset_indices),
        model=predictor,
        x_mean=x_mean,
        x_std=x_std,
        clean_test_mse=clean_test_mse,
    )
    return bundle


def train_standard_predictors(
    config: MinimaxConfig,
    *,
    run_id: int,
) -> tuple[list[PredictorBundle], NonlinearStableBlanketSCM]:
    configure_torch(config.torch_num_threads)
    torch.manual_seed(run_id)
    scm = NonlinearStableBlanketSCM(
        device=config.device,
        intervene_on_x1=config.intervene_on_x1,
        include_x6=config.include_x6,
    )
    x_train, y_train, x_val, y_val, x_test, y_test = _sample_clean_datasets(config, scm, run_id)
    sets = {
        "parents": scm.graph_sets.parents,
        "stable_blanket": scm.graph_sets.stable_blanket,
        "all_variables": scm.graph_sets.all_variables,
    }
    predictors: list[PredictorBundle] = []
    for name, subset in sets.items():
        if name not in config.methods:
            continue
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
                batch_size=config.minimax_batch_size,
                max_epochs=max(1, config.minimax_steps),
                patience=max(1, min(20, config.minimax_steps)),
                weight_decay=config.predictor_weight_decay,
            )
        )
    return predictors, scm


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["method", "sweep_value"])
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
    return summary.sort_values(["sweep_value", "method"])


def save_minimax_plot(results: pd.DataFrame, output_dir: str | Path, attack_mode: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_label = "Intervention bound" if attack_mode == "bound" else "Adversary cost"
    filename = "minimax_all_variables_vs_bound.png" if attack_mode == "bound" else "minimax_all_variables_vs_cost.png"

    plt.figure(figsize=(7, 4.5))
    present_methods = [method for method in METHOD_ORDER if method in results["method"].unique()]
    for method in present_methods:
        sub_results = results[results["method"] == method]
        if sub_results.empty:
            continue
        stats = sub_results.groupby(["sweep_value"])["attacked_test_mse"].agg(["mean", "std", "count"]).reset_index()
        stats["std"] = stats["std"].fillna(0.0)
        stats["sem"] = stats["std"] / stats["count"].clip(lower=1) ** 0.5
        stats["ci95_radius"] = 1.96 * stats["sem"]
        plt.plot(stats["sweep_value"], stats["mean"], marker="o", label=method)
        plt.fill_between(
            stats["sweep_value"],
            stats["mean"] - stats["ci95_radius"],
            stats["mean"] + stats["ci95_radius"],
            alpha=0.2,
        )
        plt.axhline(sub_results["clean_test_mse"].mean(), linestyle="--", alpha=0.25)
    plt.xlabel(x_label)
    plt.ylabel("Test MSE under fresh best-response attack")
    plt.title("Minimax vs standard predictors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180)
    plt.close()


def run_minimax_experiment(config: MinimaxConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    configure_torch(config.torch_num_threads)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_name, sweep_values = _sweep_values(config)

    rows: list[MinimaxResultRow] = []
    total_tasks = config.num_runs * len(sweep_values)
    progress = tqdm(total=total_tasks, desc="Minimax experiment", unit="run")

    for run_id in range(config.num_runs):
        standard_predictors, standard_scm = train_standard_predictors(config, run_id=run_id)
        trained_minimax_predictor: PredictorBundle | None = None
        trained_learned_minimax_predictor: PredictorBundle | None = None
        if "minimax_all_variables" in config.methods and config.attack_mode == "bound" and config.train_intervention_bound is not None:
            trained_minimax_predictor = train_minimax_predictor(
                config,
                run_id=run_id,
                bound=float(config.train_intervention_bound),
                cost=None,
            )
        if (
            "minimax_learned_scm_all_variables" in config.methods
            and config.attack_mode == "bound"
            and config.train_intervention_bound is not None
        ):
            trained_learned_minimax_predictor = train_learned_scm_minimax_predictor(
                config,
                run_id=run_id,
                bound=float(config.train_intervention_bound),
                cost=None,
            )
        for sweep_value in sweep_values:
            bound = float(sweep_value if sweep_name == "bound" else config.max_perturbation_bound)
            cost = float(sweep_value) if sweep_name == "cost" else None

            if "minimax_all_variables" in config.methods:
                predictor = trained_minimax_predictor
                if predictor is None:
                    predictor = train_minimax_predictor(config, run_id=run_id, bound=bound, cost=cost)
                attack_result = optimize_attack(
                    scm=standard_scm,
                    predictor=predictor,
                    bound=bound,
                    cost=cost,
                    intervene_on_x1=config.intervene_on_x1,
                    objective="mse",
                    restarts=config.attack_restarts,
                    steps=config.attack_steps,
                    batch_size=config.attack_batch_size,
                    hidden_dim=config.attack_hidden_dim,
                    lr=config.attack_lr,
                    eval_size=config.attack_eval_size,
                    seed=run_id,
                )
                rows.append(
                    MinimaxResultRow(
                        run_id=run_id,
                        method="minimax_all_variables",
                        sweep_value=float(sweep_value),
                        bound=float(bound),
                        cost=float(cost or 0.0),
                        clean_test_mse=float(predictor.clean_test_mse),
                        attacked_test_mse=float(attack_result.attacked_test_mse),
                        attack_objective_value=float(attack_result.best_objective_value),
                        regularized_attack_value=float(attack_result.best_regularized_value),
                        intervention_strength=float(attack_result.intervention_strength),
                    )
                )
            if "minimax_learned_scm_all_variables" in config.methods:
                predictor = trained_learned_minimax_predictor
                if predictor is None:
                    predictor = train_learned_scm_minimax_predictor(config, run_id=run_id, bound=bound, cost=cost)
                attack_result = optimize_attack(
                    scm=standard_scm,
                    predictor=predictor,
                    bound=bound,
                    cost=cost,
                    intervene_on_x1=config.intervene_on_x1,
                    objective="mse",
                    restarts=config.attack_restarts,
                    steps=config.attack_steps,
                    batch_size=config.attack_batch_size,
                    hidden_dim=config.attack_hidden_dim,
                    lr=config.attack_lr,
                    eval_size=config.attack_eval_size,
                    seed=run_id,
                )
                rows.append(
                    MinimaxResultRow(
                        run_id=run_id,
                        method="minimax_learned_scm_all_variables",
                        sweep_value=float(sweep_value),
                        bound=float(bound),
                        cost=float(cost or 0.0),
                        clean_test_mse=float(predictor.clean_test_mse),
                        attacked_test_mse=float(attack_result.attacked_test_mse),
                        attack_objective_value=float(attack_result.best_objective_value),
                        regularized_attack_value=float(attack_result.best_regularized_value),
                        intervention_strength=float(attack_result.intervention_strength),
                    )
                )
            for baseline_predictor in standard_predictors:
                baseline_attack_result = optimize_attack(
                    scm=standard_scm,
                    predictor=baseline_predictor,
                    bound=bound,
                    cost=cost,
                    intervene_on_x1=config.intervene_on_x1,
                    objective="mse",
                    restarts=config.attack_restarts,
                    steps=config.attack_steps,
                    batch_size=config.attack_batch_size,
                    hidden_dim=config.attack_hidden_dim,
                    lr=config.attack_lr,
                    eval_size=config.attack_eval_size,
                    seed=run_id,
                )
                rows.append(
                    MinimaxResultRow(
                        run_id=run_id,
                        method=baseline_predictor.name,
                        sweep_value=float(sweep_value),
                        bound=float(bound),
                        cost=float(cost or 0.0),
                        clean_test_mse=float(baseline_predictor.clean_test_mse),
                        attacked_test_mse=float(baseline_attack_result.attacked_test_mse),
                        attack_objective_value=float(baseline_attack_result.best_objective_value),
                        regularized_attack_value=float(baseline_attack_result.best_regularized_value),
                        intervention_strength=float(baseline_attack_result.intervention_strength),
                    )
                )
            progress.update(1)
            progress.set_postfix(run=run_id, **{sweep_name: float(sweep_value)})

    progress.close()
    results = pd.DataFrame([asdict(r) for r in rows])
    summary = summarize_results(results)
    results.to_csv(output_dir / "minimax_results_per_run.csv", index=False)
    summary.to_csv(output_dir / "minimax_results_summary.csv", index=False)
    save_minimax_plot(results, output_dir, config.attack_mode)
    pd.DataFrame([asdict(config)]).to_json(output_dir / "minimax_config.json", orient="records", indent=2)
    return results, summary


def main() -> None:
    args = parse_args()
    config = MinimaxConfig(
        output_dir=args.output_dir,
        device=args.device,
        torch_num_threads=args.torch_num_threads,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        predictor_hidden_dim=args.predictor_hidden_dim,
        predictor_depth=args.predictor_depth,
        predictor_lr=args.predictor_lr,
        predictor_weight_decay=args.predictor_weight_decay,
        attack_hidden_dim=args.attack_hidden_dim,
        attack_lr=args.attack_lr,
        attack_mode=args.attack_mode,
        intervene_on_x1=not args.disable_x1_intervention,
        include_x6=args.include_x6,
        train_intervention_bound=args.train_intervention_bound,
        bounds=tuple(args.bounds),
        costs=tuple(args.costs),
        max_perturbation_bound=args.max_perturbation_bound,
        minimax_steps=args.minimax_steps,
        predictor_steps_per_iter=args.predictor_steps_per_iter,
        adversary_steps_per_iter=args.adversary_steps_per_iter,
        adversary_reinit_interval=args.adversary_reinit_interval,
        minimax_batch_size=args.minimax_batch_size,
        minimax_validation_interval=args.minimax_validation_interval,
        val_attack_restarts=args.val_attack_restarts,
        val_attack_steps=args.val_attack_steps,
        val_attack_batch_size=args.val_attack_batch_size,
        attack_restarts=args.attack_restarts,
        attack_steps=args.attack_steps,
        attack_batch_size=args.attack_batch_size,
        attack_eval_size=args.attack_eval_size,
        num_runs=args.num_runs,
        methods=tuple(args.methods),
    )
    _, summary = run_minimax_experiment(config)
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {config.output_dir}")


if __name__ == "__main__":
    main()
