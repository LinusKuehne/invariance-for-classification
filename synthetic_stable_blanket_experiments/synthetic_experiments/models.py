from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(last, hidden_dim), nn.ReLU()])
            last = hidden_dim
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FixedLinearRegressor(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor | None = None) -> None:
        super().__init__()
        input_dim = int(weight.numel())
        self.linear = nn.Linear(input_dim, 1, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(weight.reshape(1, input_dim))
            if bias is None:
                self.linear.bias.zero_()
            else:
                self.linear.bias.copy_(bias.reshape(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


@dataclass
class PredictorBundle:
    name: str
    subset_indices: tuple[int, ...]
    model: nn.Module
    x_mean: Tensor
    x_std: Tensor
    clean_test_mse: float

    def select_and_standardize(self, x_full: Tensor) -> Tensor:
        x = x_full[:, self.subset_indices]
        return (x - self.x_mean) / self.x_std

    @torch.no_grad()
    def predict(self, x_full: Tensor) -> Tensor:
        self.model.eval()
        return self.model(self.select_and_standardize(x_full))

    @torch.no_grad()
    def mse(self, x_full: Tensor, y: Tensor) -> float:
        return torch.mean((y - self.predict(x_full)) ** 2).item()


def train_predictor(
    *,
    name: str,
    subset_indices: Sequence[int],
    x_train_full: Tensor,
    y_train: Tensor,
    x_val_full: Tensor,
    y_val: Tensor,
    x_test_full: Tensor,
    y_test: Tensor,
    hidden_dim: int,
    depth: int,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    weight_decay: float,
) -> PredictorBundle:
    subset_indices = tuple(int(i) for i in subset_indices)
    x_train = x_train_full[:, subset_indices]
    x_val = x_val_full[:, subset_indices]
    x_test = x_test_full[:, subset_indices]

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train_std = (x_train - x_mean) / x_std
    x_val_std = (x_val - x_mean) / x_std
    x_test_std = (x_test - x_mean) / x_std

    model = MLPRegressor(len(subset_indices), hidden_dim=hidden_dim, depth=depth).to(x_train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_train_std, y_train), batch_size=batch_size, shuffle=True, num_workers=0)

    best_state = None
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
            val = criterion(model(x_val_std), y_val).item()

        if val < best_val - 1e-8:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        clean_test_mse = criterion(model(x_test_std), y_test).item()

    return PredictorBundle(
        name=name,
        subset_indices=subset_indices,
        model=model,
        x_mean=x_mean,
        x_std=x_std,
        clean_test_mse=clean_test_mse,
    )


def build_oracle_predictor(
    *,
    name: str,
    subset_indices: Sequence[int],
    standardized_weight: Tensor,
    x_mean: Tensor,
    x_std: Tensor,
    x_test_full: Tensor,
    y_test: Tensor,
) -> PredictorBundle:
    subset_indices = tuple(int(i) for i in subset_indices)
    model = FixedLinearRegressor(weight=standardized_weight.to(x_test_full.device)).to(x_test_full.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    x_test = x_test_full[:, subset_indices]
    x_test_std = (x_test - x_mean) / x_std
    with torch.no_grad():
        clean_test_mse = torch.mean((y_test - model(x_test_std)) ** 2).item()

    return PredictorBundle(
        name=name,
        subset_indices=subset_indices,
        model=model,
        x_mean=x_mean,
        x_std=x_std,
        clean_test_mse=clean_test_mse,
    )
