"""
IRM implementation closely following the InvarianceUnitTests repository
(Facebook Research, Arjovsky et al.).

Key differences from the neural_net.py IRMClassifier:
  - Linear model (no hidden layers), matching InvarianceUnitTests
  - IRMLayer wrapper with dummy_mul / dummy_sum parameters
  - Penalty: sum_e ||grad_{dummy} L_e||^2  (gradient w.r.t. dummies)
  - Objective: (1 - lambda) * avg_loss + lambda * penalty
  - Optimizer only updates real parameters (not dummies)

Wrapped in a sklearn-like interface for use in evaluate_all.py.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss
from torch.autograd import grad

warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────────────────────────────────────
# IRMLayer  (from InvarianceUnitTests/scripts/models.py)
# ──────────────────────────────────────────────────────────────────────────────


class IRMLayer(nn.Module):
    """
    Wraps a layer with dummy multiply-by-one / add-zero parameters
    so that we can take gradients w.r.t. the dummies for the IRM penalty.

    Taken directly from InvarianceUnitTests.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.dummy_mul = nn.Parameter(torch.Tensor([1.0]))
        self.dummy_sum = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x) * self.dummy_mul + self.dummy_sum


def _find_parameters(
    network: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Separate real parameters from dummy parameters."""
    parameters: list[nn.Parameter] = []
    dummies: list[nn.Parameter] = []
    for name, param in network.named_parameters():
        if "dummy" in name:
            dummies.append(param)
        else:
            parameters.append(param)
    return parameters, dummies


# ──────────────────────────────────────────────────────────────────────────────
# hyperparams
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _Hparams:
    """Hyperparameters for a single training run."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    irm_lambda: float = 0.9
    num_iterations: int = 10000
    seed: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def _standardise(
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Per-feature z-scoring fitted on X_train only."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - mu) / sigma
    X_test_s = (X_test - mu) / sigma if X_test is not None else None
    return X_train_s, X_test_s, mu, sigma


def _split_by_env(
    X: np.ndarray,
    y: np.ndarray,
    E: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Split arrays into per-environment lists."""
    env_ids = np.sort(np.unique(E))
    X_envs = [X[E == e] for e in env_ids]
    y_envs = [y[E == e] for e in env_ids]
    return X_envs, y_envs, env_ids


def _split_train_val(
    X_envs: list[np.ndarray],
    y_envs: list[np.ndarray],
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """80/20 split within each environment."""
    rng = np.random.RandomState(seed)
    Xtr, ytr, Xva, yva = [], [], [], []
    for X_e, y_e in zip(X_envs, y_envs, strict=True):
        n = len(X_e)
        idx = rng.permutation(n)
        cut = int(n * (1 - val_frac))
        Xtr.append(X_e[idx[:cut]])
        ytr.append(y_e[idx[:cut]])
        Xva.append(X_e[idx[cut:]])
        yva.append(y_e[idx[cut:]])
    return Xtr, ytr, Xva, yva


# ──────────────────────────────────────────────────────────────────────────────
# training (follows InvarianceUnitTests logic)
# ──────────────────────────────────────────────────────────────────────────────


def _train_irm_linear(
    X_envs: list[np.ndarray],
    y_envs: list[np.ndarray],
    hparams: _Hparams,
    X_val_envs: list[np.ndarray] | None = None,
    y_val_envs: list[np.ndarray] | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> nn.Module:
    """
    Train a linear IRM model following the InvarianceUnitTests approach.

    Architecture: Linear -> IRMLayer (dummy_mul, dummy_sum)
    Penalty:      sum_e ||grad_{dummy} L_e||^2
    Objective:    (1 - lambda) * avg_loss + lambda * penalty
    Optimizer:    Adam on real parameters only (not dummies)
    """
    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)

    n_features = X_envs[0].shape[1]

    # Linear model wrapped in IRMLayer (matching InvarianceUnitTests)
    linear = nn.Linear(n_features, 1)
    network = IRMLayer(linear)
    network = network.to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    # Separate real params from dummies (only optimise real params)
    net_parameters, net_dummies = _find_parameters(network)

    optimizer = torch.optim.Adam(
        net_parameters,
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    # Convert data to tensors (full-batch, matching InvarianceUnitTests)
    env_data = []
    for X_e, y_e in zip(X_envs, y_envs, strict=True):
        xt = torch.tensor(X_e, dtype=torch.float32, device=device)
        yt = torch.tensor(y_e, dtype=torch.float32, device=device).unsqueeze(-1)
        env_data.append((xt, yt))

    for _ in range(hparams.num_iterations):
        # Compute per-environment losses and gradients w.r.t. dummies
        losses_env = []
        gradients_env = []
        for x_e, y_e in env_data:
            loss_e = loss_fn(network(x_e), y_e)
            losses_env.append(loss_e)
            gradients_env.append(grad(loss_e, net_dummies, create_graph=True))

        # Average loss across environments
        losses_avg = sum(losses_env) / len(losses_env)

        # IRM penalty: sum_e ||grad_{dummy} L_e||^2
        penalty = torch.tensor(0.0, device=device)
        for gradients_this_env in gradients_env:
            for g_env in gradients_this_env:
                penalty = penalty + g_env.pow(2).sum()

        # Objective: (1 - lambda) * avg_loss + lambda * penalty
        obj = (1 - hparams.irm_lambda) * losses_avg
        obj = obj + hparams.irm_lambda * penalty

        optimizer.zero_grad()
        obj.backward()
        optimizer.step()

    network.eval()
    return network


@torch.no_grad()
def _predict_proba_linear(
    network: nn.Module, X: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """Return P(Y=1) for each sample."""
    network.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    logits = network(Xt).squeeze(-1)
    prob = torch.sigmoid(logits).cpu().numpy()
    return prob


# ──────────────────────────────────────────────────────────────────────────────
# grid search
# ──────────────────────────────────────────────────────────────────────────────


def _build_grid() -> list[dict]:
    """Build hyperparameter grid matching InvarianceUnitTests style."""
    grid = {
        "lr": [1e-3, 5e-4, 1e-4],
        "weight_decay": [0.0, 1e-4],
        "irm_lambda": [0.5, 0.9, 0.99],
        "num_iterations": [5000, 10000],
        "seed": [0],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    return [dict(zip(keys, combo, strict=True)) for combo in combos]


def _evaluate_single_config(
    cfg: dict,
    X_tr: list[np.ndarray],
    y_tr: list[np.ndarray],
    X_va: list[np.ndarray],
    y_va: list[np.ndarray],
    device: str,
) -> tuple[dict, float, float]:
    """Train one config, return (cfg, worst_val_bce, mean_val_bce)."""
    hp = _Hparams(
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        irm_lambda=cfg["irm_lambda"],
        num_iterations=cfg["num_iterations"],
        seed=cfg["seed"],
    )

    model = _train_irm_linear(
        X_tr, y_tr, hp, X_val_envs=X_va, y_val_envs=y_va, device=device
    )

    val_bces = []
    for x_v, y_v in zip(X_va, y_va, strict=True):
        prob_v = _predict_proba_linear(model, x_v, device)
        bce_v = log_loss(y_v, np.clip(prob_v, 1e-7, 1 - 1e-7), labels=[0, 1])
        val_bces.append(bce_v)

    return cfg, float(max(val_bces)), float(np.mean(val_bces))


def _grid_search(
    X_envs: list[np.ndarray],
    y_envs: list[np.ndarray],
    val_frac: float = 0.2,
    val_seed: int = 42,
    device: str = "cpu",
    n_jobs: int = 1,
    verbose: bool = False,
) -> tuple[dict | None, list[tuple[dict, float, float]]]:
    """Grid search over IRM hyperparameters."""
    grid = _build_grid()

    X_tr, y_tr, X_va, y_va = _split_train_val(
        X_envs, y_envs, val_frac=val_frac, seed=val_seed
    )

    if verbose:
        print(f"    Grid search: {len(grid)} configs, n_jobs={n_jobs}")

    raw_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_evaluate_single_config)(cfg, X_tr, y_tr, X_va, y_va, device)
        for cfg in grid
    )
    results: list[tuple[dict, float, float]] = [r for r in raw_results if r is not None]

    best_metric = float("inf")
    best_cfg: dict | None = None
    for cfg, worst_bce, _ in results:
        if worst_bce < best_metric:
            best_metric = worst_bce
            best_cfg = cfg.copy()

    if verbose and best_cfg is not None:
        print(f"    Best config: worst_val_bce={best_metric:.4f}")
        print(f"      {best_cfg}")

    return best_cfg, results


# ──────────────────────────────────────────────────────────────────────────────
# sklearn-style classifier
# ──────────────────────────────────────────────────────────────────────────────


class IRM2Classifier(ClassifierMixin, BaseEstimator):
    """
    IRM classifier following the InvarianceUnitTests (Facebook Research)
    implementation.

    Uses a linear model with IRMLayer dummy parameters.
    The IRM penalty is computed as the squared gradient of per-environment
    losses w.r.t. the dummy parameters.

    Parameters
    ----------
    n_jobs : int, default=1
        Number of parallel workers for grid search.
    device : str, default="cpu"
        PyTorch device ("cpu" or "cuda").
    verbose : bool, default=False
        If True, print grid search progress.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_jobs: int = 1,
        device: str = "cpu",
        verbose: bool = False,
        random_state: int = 42,
    ):
        self.n_jobs = n_jobs
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

    def fit(
        self, X: np.ndarray, y: np.ndarray, environment: np.ndarray
    ) -> "IRM2Classifier":
        """
        Fit the model with hyperparameter grid search.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Binary target (0 or 1).
        environment : array-like of shape (n_samples,)
            Environment labels.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        environment = np.asarray(environment)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        # Standardise features
        X_s, _, mu, sigma = _standardise(X)
        self._mu = mu
        self._sigma = sigma

        # Split by environment
        X_envs, y_envs, _ = _split_by_env(X_s, y, environment)

        if self.verbose:
            print("  Fitting IRM 2 (InvarianceUnitTests-style)...")

        # Grid search
        best_cfg, self._grid_results = _grid_search(
            X_envs,
            y_envs,
            val_frac=0.2,
            val_seed=self.random_state,
            device=self.device,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        if best_cfg is None:
            raise RuntimeError("Grid search failed to find any valid configuration.")

        self.best_config_ = best_cfg

        # Retrain on full training data with best config
        X_tr, y_tr, X_va, y_va = _split_train_val(
            X_envs, y_envs, val_frac=0.2, seed=self.random_state
        )

        hp = _Hparams(
            lr=best_cfg["lr"],
            weight_decay=best_cfg["weight_decay"],
            irm_lambda=best_cfg["irm_lambda"],
            num_iterations=best_cfg["num_iterations"],
            seed=best_cfg["seed"],
        )

        self._model = _train_irm_linear(
            X_tr,
            y_tr,
            hp,
            X_val_envs=X_va,
            y_val_envs=y_va,
            device=self.device,
            verbose=self.verbose,
        )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates [P(Y=0), P(Y=1)] per sample."""
        X = np.asarray(X, dtype=np.float64)
        X_s = (X - self._mu) / self._sigma
        prob_1 = _predict_proba_linear(self._model, X_s, self.device)
        return np.column_stack([1 - prob_1, prob_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
