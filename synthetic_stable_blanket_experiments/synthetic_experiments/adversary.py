from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .models import PredictorBundle
from .scm import AttackMechanisms, NonlinearStableBlanketSCM

AttackObjective = Literal["signed_error", "mse"]


@dataclass
class AttackResult:
    objective_name: str
    bound: float
    cost: float | None
    best_objective_value: float
    best_regularized_value: float
    intervention_strength: float
    attacked_test_mse: float
    attack: AttackMechanisms


def _objective(y: Tensor, pred: Tensor, objective: AttackObjective) -> Tensor:
    if objective == "signed_error":
        return torch.mean(y - pred)
    if objective == "mse":
        return torch.mean((y - pred) ** 2)
    raise ValueError(objective)


def _intervention_strength(delta_x1: Tensor, delta_x4: Tensor) -> Tensor:
    return torch.mean(delta_x1.square() + delta_x4.square())


def optimize_attack(
    *,
    scm: NonlinearStableBlanketSCM,
    predictor: PredictorBundle,
    bound: float,
    cost: float | None,
    intervene_on_x1: bool,
    simple: bool,
    objective: AttackObjective,
    restarts: int,
    steps: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    eval_size: int,
    seed: int,
) -> AttackResult:
    # Predictor is frozen.
    predictor.model.eval()
    for p in predictor.model.parameters():
        p.requires_grad_(False)

    best_attack: AttackMechanisms | None = None
    best_value: float | None = None
    best_regularized_value: float | None = None

    for restart in range(restarts):
        torch.manual_seed(seed + 1000 * restart)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 1000 * restart)
        attack = AttackMechanisms(
            bound=bound,
            hidden_dim=hidden_dim,
            intervene_on_x1=intervene_on_x1,
            simple=simple,
        ).to(scm.device)
        optimizer = torch.optim.Adam(attack.parameters(), lr=lr)
        generator = torch.Generator(device=scm.device)
        generator.manual_seed(seed + 1000 * restart + 123)

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            x_adv, y_adv, delta_x1, delta_x4 = scm.sample_with_intervention_info(
                batch_size, attack=attack, generator=generator
            )
            pred = predictor.model(predictor.select_and_standardize(x_adv))
            value = _objective(y_adv, pred, objective)
            strength = _intervention_strength(delta_x1, delta_x4)
            if cost is None:
                loss = value if objective == "signed_error" else -value
            elif objective == "signed_error":
                loss = value + cost * strength
            else:
                loss = -value + cost * strength
            loss.backward()
            optimizer.step()

        eval_gen = torch.Generator(device=scm.device)
        eval_gen.manual_seed(seed + 5000 + restart)
        with torch.no_grad():
            x_eval, y_eval, delta_x1_eval, delta_x4_eval = (
                scm.sample_with_intervention_info(
                    eval_size, attack=attack, generator=eval_gen
                )
            )
            pred_eval = predictor.model(predictor.select_and_standardize(x_eval))
            value_eval = _objective(y_eval, pred_eval, objective).item()
            strength_eval = _intervention_strength(delta_x1_eval, delta_x4_eval).item()

        regularized_value_eval = value_eval
        if cost is not None:
            regularized_value_eval = (
                value_eval - cost * strength_eval
                if objective == "mse"
                else value_eval + cost * strength_eval
            )

        better = best_value is None
        if objective == "signed_error" and best_regularized_value is not None:
            better = regularized_value_eval < best_regularized_value
        elif objective == "mse" and best_regularized_value is not None:
            better = regularized_value_eval > best_regularized_value

        if better:
            best_value = value_eval
            best_regularized_value = regularized_value_eval
            best_attack = AttackMechanisms(
                bound=bound,
                hidden_dim=hidden_dim,
                intervene_on_x1=intervene_on_x1,
                simple=simple,
            ).to(scm.device)
            best_attack.load_state_dict(attack.state_dict())

    assert (
        best_attack is not None
        and best_value is not None
        and best_regularized_value is not None
    )
    test_gen = torch.Generator(device=scm.device)
    test_gen.manual_seed(seed + 99999)
    with torch.no_grad():
        x_test_adv, y_test_adv, delta_x1_test, delta_x4_test = (
            scm.sample_with_intervention_info(
                eval_size, attack=best_attack, generator=test_gen
            )
        )
        attacked_test_mse = torch.mean(
            (y_test_adv - predictor.predict(x_test_adv)) ** 2
        ).item()
        intervention_strength = _intervention_strength(
            delta_x1_test, delta_x4_test
        ).item()

    return AttackResult(
        objective_name=objective,
        bound=float(bound),
        cost=cost,
        best_objective_value=float(best_value),
        best_regularized_value=float(best_regularized_value),
        intervention_strength=float(intervention_strength),
        attacked_test_mse=float(attacked_test_mse),
        attack=best_attack,
    )
