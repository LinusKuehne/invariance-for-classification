from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor, nn

NoiseDistribution = Literal["gaussian", "student_t"]


def _sample_noise(
    n: int,
    dim: int,
    *,
    device: torch.device,
    distribution: NoiseDistribution,
    student_t_df: int,
    generator: Optional[torch.Generator],
) -> Tensor:
    randn_kwargs = {"device": device}
    if generator is not None:
        randn_kwargs["generator"] = generator
    if distribution == "gaussian":
        return torch.randn(n, dim, **randn_kwargs)
    if distribution == "student_t":
        if student_t_df <= 2:
            raise ValueError("student_t_df must be greater than 2 for finite variance.")
        numerator = torch.randn(n, dim, **randn_kwargs)
        denominator_samples = torch.randn(n, dim, student_t_df, **randn_kwargs)
        chi_square = denominator_samples.square().sum(dim=-1)
        student_t = numerator / torch.sqrt(chi_square / float(student_t_df))
        variance_scale = ((student_t_df - 2.0) / float(student_t_df)) ** 0.5
        return variance_scale * student_t
    raise ValueError(f"Unknown noise distribution: {distribution}")


@dataclass(frozen=True)
class GraphSets:
    parents: tuple[int, ...] = (0, 1)
    stable_blanket: tuple[int, ...] = (0, 1, 2)
    all_variables: tuple[int, ...] = (0, 1, 2, 3, 4)
    mutable: tuple[int, ...] = (0, 3)


class AttackNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, bound: float) -> None:
        super().__init__()
        self.bound = float(bound)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.bound * torch.tanh(self.net(inputs))


class ConstantAttack(nn.Module):
    def __init__(self, bound: float) -> None:
        super().__init__()
        self.bound = float(bound)
        self.raw_delta = nn.Parameter(torch.zeros(1, 1))

    def forward(self, reference: Tensor) -> Tensor:
        delta = self.bound * torch.tanh(self.raw_delta)
        return delta.expand_as(reference)


class AttackMechanisms(nn.Module):
    def __init__(
        self,
        bound: float,
        hidden_dim: int = 64,
        intervene_on_x1: bool = True,
        simple: bool = False,
        x4_uses_x1_x3: bool = False,
    ) -> None:
        super().__init__()
        self.bound = float(bound)
        self.intervene_on_x1 = intervene_on_x1
        self.simple = simple
        self.x4_uses_x1_x3 = x4_uses_x1_x3
        if simple:
            self.x1_net = ConstantAttack(bound) if intervene_on_x1 else None
            self.x4_net = ConstantAttack(bound)
        else:
            self.x1_net = (
                AttackNetwork(2, hidden_dim, bound) if intervene_on_x1 else None
            )
            x4_input_dim = 5 if x4_uses_x1_x3 else 3
            self.x4_net = AttackNetwork(x4_input_dim, hidden_dim, bound)

    def perturb_x1(self, x2: Tensor, noise: Tensor) -> Tensor:
        if self.x1_net is None:
            return torch.zeros_like(x2)
        if self.simple:
            return self.x1_net(x2)
        return self.x1_net(torch.cat([x2, noise], dim=1))

    def perturb_x4(
        self,
        y: Tensor,
        x2: Tensor,
        noise: Tensor,
        x1: Tensor | None = None,
        x3: Tensor | None = None,
    ) -> Tensor:
        if self.simple:
            return self.x4_net(y)
        pieces = [y, x2, noise]
        if self.x4_uses_x1_x3:
            if x1 is None or x3 is None:
                raise ValueError(
                    "x1 and x3 are required when x4_uses_x1_x3 is enabled."
                )
            pieces.extend([x1, x3])
        return self.x4_net(torch.cat(pieces, dim=1))


class NonlinearStableBlanketSCM:
    """Nonlinear SCM with stable blanket {X1, X2, X3} and direct intervention on X1 and X4."""

    def __init__(
        self,
        device: str = "cpu",
        intervene_on_x1: bool = True,
        include_x6: bool = False,
        noise_distribution: NoiseDistribution = "gaussian",
        student_t_df: int = 3,
    ) -> None:
        self.device = torch.device(device)
        self.intervene_on_x1 = intervene_on_x1
        self.include_x6 = include_x6
        self.noise_distribution = noise_distribution
        self.student_t_df = int(student_t_df)
        mutable = (0, 3) if intervene_on_x1 else (3,)
        all_variables = (0, 1, 2, 3, 4, 5) if include_x6 else (0, 1, 2, 3, 4)
        self.graph_sets = GraphSets(all_variables=all_variables, mutable=mutable)

    def _noise(self, n: int, generator: Optional[torch.Generator]) -> Tensor:
        return _sample_noise(
            n,
            1,
            device=self.device,
            distribution=self.noise_distribution,
            student_t_df=self.student_t_df,
            generator=generator,
        )

    @staticmethod
    def _clean_x1(x2: Tensor, noise: Tensor) -> Tensor:
        return 0.8 * torch.tanh(x2) + 0.6 * x2 + 0.3 * noise

    @staticmethod
    def _clean_x4_from_parents(y: Tensor, x2: Tensor, noise: Tensor) -> Tensor:
        return 2.1 * torch.tanh(y) + 0.9 * y + 0.55 * x2 + 0.15 * y * x2 + 0.2 * noise

    @staticmethod
    def _clean_x5(x4: Tensor, noise: Tensor) -> Tensor:
        return 1.5 * torch.tanh(x4) + 0.75 * x4 + 0.35 * x4.square() + 0.2 * noise

    @staticmethod
    def _clean_x6(x4: Tensor, y: Tensor, noise: Tensor) -> Tensor:
        return 1.1 * torch.tanh(x4) + 0.6 * y + 0.2 * x4 * y + 0.15 * noise

    def sample(
        self,
        n: int,
        attack: Optional[AttackMechanisms] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, Tensor]:
        x2 = self._noise(n, generator)
        eps1 = self._noise(n, generator)
        eps_y = self._noise(n, generator)
        eps3 = self._noise(n, generator)
        eps4 = self._noise(n, generator)
        eps5 = self._noise(n, generator)
        eps6 = self._noise(n, generator)

        x1 = self._clean_x1(x2, eps1)
        if attack is not None and self.intervene_on_x1:
            x1 = x1 + attack.perturb_x1(x2, eps1)

        y = torch.sin(x1) + 0.8 * x2 + 0.5 * x1 * x2 + 0.4 * eps_y
        x3 = torch.tanh(1.2 * y) + 0.45 * y.square() + 0.5 * eps3
        x4 = self._clean_x4_from_parents(y, x2, eps4)
        if attack is not None:
            x4 = x4 + attack.perturb_x4(y, x2, eps4, x1=x1, x3=x3)

        x5 = self._clean_x5(x4, eps5)
        pieces = [x1, x2, x3, x4, x5]
        if self.include_x6:
            x6 = self._clean_x6(x4, y, eps6)
            pieces.append(x6)

        x = torch.cat(pieces, dim=1)
        return x, y

    def sample_with_intervention_info(
        self,
        n: int,
        attack: Optional[AttackMechanisms] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x2 = self._noise(n, generator)
        eps1 = self._noise(n, generator)
        eps_y = self._noise(n, generator)
        eps3 = self._noise(n, generator)
        eps4 = self._noise(n, generator)
        eps5 = self._noise(n, generator)
        eps6 = self._noise(n, generator)

        clean_x1 = self._clean_x1(x2, eps1)
        delta_x1 = torch.zeros_like(clean_x1)
        if attack is not None and self.intervene_on_x1:
            delta_x1 = attack.perturb_x1(x2, eps1)
        x1 = clean_x1 + delta_x1

        y = torch.sin(x1) + 0.8 * x2 + 0.5 * x1 * x2 + 0.4 * eps_y
        x3 = torch.tanh(1.2 * y) + 0.45 * y.square() + 0.5 * eps3

        clean_x4 = self._clean_x4_from_parents(y, x2, eps4)
        delta_x4 = torch.zeros_like(clean_x4)
        if attack is not None:
            delta_x4 = attack.perturb_x4(y, x2, eps4, x1=x1, x3=x3)
        x4 = clean_x4 + delta_x4

        x5 = self._clean_x5(x4, eps5)
        pieces = [x1, x2, x3, x4, x5]
        if self.include_x6:
            x6 = self._clean_x6(x4, y, eps6)
            pieces.append(x6)

        x = torch.cat(pieces, dim=1)
        return x, y, delta_x1, delta_x4


class LinearGaussianStableBlanketSCM:
    """Linear-Gaussian SCM with exact oracle conditional expectations."""

    def __init__(
        self,
        device: str = "cpu",
        intervene_on_x1: bool = True,
        include_x6: bool = False,
        noise_distribution: NoiseDistribution = "gaussian",
        student_t_df: int = 3,
    ) -> None:
        self.device = torch.device(device)
        self.intervene_on_x1 = intervene_on_x1
        self.include_x6 = include_x6
        self.noise_distribution = noise_distribution
        self.student_t_df = int(student_t_df)
        mutable = (0, 3) if intervene_on_x1 else (3,)
        all_variables = (0, 1, 2, 3, 4, 5) if include_x6 else (0, 1, 2, 3, 4)
        self.graph_sets = GraphSets(all_variables=all_variables, mutable=mutable)

    def _noise(self, n: int, generator: Optional[torch.Generator]) -> Tensor:
        return _sample_noise(
            n,
            1,
            device=self.device,
            distribution=self.noise_distribution,
            student_t_df=self.student_t_df,
            generator=generator,
        )

    @staticmethod
    def _clean_x1(x2: Tensor, noise: Tensor) -> Tensor:
        return 1.1 * x2 + 0.3 * noise

    @staticmethod
    def _clean_y(x1: Tensor, x2: Tensor, noise: Tensor) -> Tensor:
        return 1.0 * x1 + 0.8 * x2 + 0.4 * noise

    @staticmethod
    def _clean_x3(y: Tensor, noise: Tensor) -> Tensor:
        return 1.2 * y + 0.5 * noise

    @staticmethod
    def _clean_x4_from_parents(y: Tensor, x2: Tensor, noise: Tensor) -> Tensor:
        return 1.4 * y + 0.55 * x2 + 0.2 * noise

    @staticmethod
    def _clean_x5(x4: Tensor, noise: Tensor) -> Tensor:
        return 1.3 * x4 + 0.2 * noise

    @staticmethod
    def _clean_x6(x4: Tensor, y: Tensor, noise: Tensor) -> Tensor:
        return 1.1 * x4 + 0.6 * y + 0.15 * noise

    def sample(
        self,
        n: int,
        attack: Optional[AttackMechanisms] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, Tensor]:
        x, y, _, _ = self.sample_with_intervention_info(
            n, attack=attack, generator=generator
        )
        return x, y

    def sample_with_intervention_info(
        self,
        n: int,
        attack: Optional[AttackMechanisms] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x2 = self._noise(n, generator)
        eps1 = self._noise(n, generator)
        eps_y = self._noise(n, generator)
        eps3 = self._noise(n, generator)
        eps4 = self._noise(n, generator)
        eps5 = self._noise(n, generator)
        eps6 = self._noise(n, generator)

        clean_x1 = self._clean_x1(x2, eps1)
        delta_x1 = torch.zeros_like(clean_x1)
        if attack is not None and self.intervene_on_x1:
            delta_x1 = attack.perturb_x1(x2, eps1)
        x1 = clean_x1 + delta_x1

        y = self._clean_y(x1, x2, eps_y)
        x3 = self._clean_x3(y, eps3)

        clean_x4 = self._clean_x4_from_parents(y, x2, eps4)
        delta_x4 = torch.zeros_like(clean_x4)
        if attack is not None:
            delta_x4 = attack.perturb_x4(y, x2, eps4, x1=x1, x3=x3)
        x4 = clean_x4 + delta_x4

        x5 = self._clean_x5(x4, eps5)
        pieces = [x1, x2, x3, x4, x5]
        if self.include_x6:
            x6 = self._clean_x6(x4, y, eps6)
            pieces.append(x6)

        x = torch.cat(pieces, dim=1)
        return x, y, delta_x1, delta_x4

    def conditional_mean_params(
        self, subset_indices: tuple[int, ...]
    ) -> tuple[Tensor, Tensor, Tensor]:
        cov_xx, cov_xy = self._joint_covariances()
        subset = torch.tensor(subset_indices, dtype=torch.long, device=self.device)
        cov_subset = cov_xx.index_select(0, subset).index_select(1, subset)
        cov_y_subset = cov_xy.index_select(0, subset)
        raw_weight = torch.linalg.solve(cov_subset, cov_y_subset).squeeze(-1)
        subset_var = torch.diagonal(cov_subset)
        x_std = subset_var.sqrt().unsqueeze(0)
        standardized_weight = (raw_weight * x_std.squeeze(0)).unsqueeze(0)
        x_mean = torch.zeros(1, len(subset_indices), device=self.device)
        return x_mean, x_std, standardized_weight

    def _joint_covariances(self) -> tuple[Tensor, Tensor]:
        noise_scales = {
            "x2": 1.0,
            "x1": 0.3,
            "y": 0.4,
            "x3": 0.5,
            "x4": 0.2,
            "x5": 0.2,
            "x6": 0.15,
        }

        names = ["x2", "x1", "y", "x3", "x4", "x5"]
        if self.include_x6:
            names.append("x6")
        name_to_idx = {name: idx for idx, name in enumerate(names)}

        dim = len(names)
        system = torch.eye(dim, device=self.device)
        noise_cov = torch.zeros(dim, dim, device=self.device)
        for name in names:
            noise_cov[name_to_idx[name], name_to_idx[name]] = noise_scales[name] ** 2

        system[name_to_idx["x1"], name_to_idx["x2"]] = -1.1
        system[name_to_idx["y"], name_to_idx["x1"]] = -1.0
        system[name_to_idx["y"], name_to_idx["x2"]] = -0.8
        system[name_to_idx["x3"], name_to_idx["y"]] = -1.2
        system[name_to_idx["x4"], name_to_idx["y"]] = -1.4
        system[name_to_idx["x4"], name_to_idx["x2"]] = -0.55
        system[name_to_idx["x5"], name_to_idx["x4"]] = -1.3
        if self.include_x6:
            system[name_to_idx["x6"], name_to_idx["x4"]] = -1.1
            system[name_to_idx["x6"], name_to_idx["y"]] = -0.6

        transform = torch.linalg.inv(system)
        covariance = transform @ noise_cov @ transform.T
        observed_names = ["x1", "x2", "x3", "x4", "x5"]
        if self.include_x6:
            observed_names.append("x6")
        observed_idx = torch.tensor(
            [name_to_idx[name] for name in observed_names],
            dtype=torch.long,
            device=self.device,
        )
        cov_xx = covariance.index_select(0, observed_idx).index_select(1, observed_idx)
        cov_xy = covariance.index_select(0, observed_idx).index_select(
            1, torch.tensor([name_to_idx["y"]], dtype=torch.long, device=self.device)
        )
        return cov_xx, cov_xy
