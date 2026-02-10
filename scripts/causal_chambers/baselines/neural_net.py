"""
Neural network baselines: IRM, V-REx, and ERM with sklearn-like interface.

Standalone, self-contained implementation of:
- IRM (Arjovsky et al., 2019)
- V-REx (Krueger et al., 2020)
- ERM (standard neural network baseline)

Designed for low-dimensional tabular data with environment labels.

These are NOT part of the main package — they exist only for comparison
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────────────────────────────────────
# MLP
# ──────────────────────────────────────────────────────────────────────────────


class TabularMLP(nn.Module):
    """Small MLP for low-dimensional tabular binary classification."""

    def __init__(
        self,
        n_features: int,
        hidden_width: int = 32,
        depth: int = 2,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_width))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ──────────────────────────────────────────────────────────────────────────────
# IRM penalty
# ──────────────────────────────────────────────────────────────────────────────


def irm_penalty(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""IRMv1 penalty."""
    scale = torch.ones(1, device=logits.device, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logits * scale, y)
    (grad,) = torch.autograd.grad(loss, scale, create_graph=True)
    return grad.pow(2)


# ──────────────────────────────────────────────────────────────────────────────
# hyperparams
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _Hparams:
    """Hyperparameters for a single training run."""

    hidden_width: int = 32
    depth: int = 2
    dropout: float = 0.0
    use_batchnorm: bool = True
    lr: float = 1e-3
    weight_decay: float = 0.0
    irm_lambda: float = 0.0
    vrex_lambda: float = 0.0
    anneal_iters: int = 100
    n_epochs: int = 500
    patience: int = 100
    batch_size: int | None = None
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
# training loop
# ──────────────────────────────────────────────────────────────────────────────


def _make_batches(
    Xt: list[torch.Tensor],
    yt: list[torch.Tensor],
    batch_size: int,
    rng: torch.Generator,
) -> list[tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """Create aligned mini-batches across all environments."""
    n_envs = len(Xt)
    perms = [torch.randperm(len(Xt[e]), generator=rng) for e in range(n_envs)]
    n_batches = min(len(p) // batch_size for p in perms)
    if n_batches == 0:
        return [([x for x in Xt], [y for y in yt])]
    batches = []
    for b in range(n_batches):
        xb = [
            Xt[e][perms[e][b * batch_size : (b + 1) * batch_size]]
            for e in range(n_envs)
        ]
        yb = [
            yt[e][perms[e][b * batch_size : (b + 1) * batch_size]]
            for e in range(n_envs)
        ]
        batches.append((xb, yb))
    return batches


def _train_model(
    X_envs: list[np.ndarray],
    y_envs: list[np.ndarray],
    hparams: _Hparams,
    X_val_envs: list[np.ndarray] | None = None,
    y_val_envs: list[np.ndarray] | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> TabularMLP:
    """Train an IRM/V-REx/ERM model."""
    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    rng = torch.Generator().manual_seed(hparams.seed)

    n_features = X_envs[0].shape[1]
    model = TabularMLP(
        n_features=n_features,
        hidden_width=hparams.hidden_width,
        depth=hparams.depth,
        dropout=hparams.dropout,
        use_batchnorm=hparams.use_batchnorm,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay
    )

    Xt = [torch.tensor(x, dtype=torch.float32, device=device) for x in X_envs]
    yt = [torch.tensor(y, dtype=torch.float32, device=device) for y in y_envs]

    has_val = X_val_envs is not None and y_val_envs is not None
    if has_val:
        assert X_val_envs is not None and y_val_envs is not None
        Xv = [torch.tensor(x, dtype=torch.float32, device=device) for x in X_val_envs]
        yv = [torch.tensor(y, dtype=torch.float32, device=device) for y in y_val_envs]
    else:
        Xv, yv = [], []

    best_val_metric = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    annealing_done = hparams.anneal_iters == 0
    use_minibatch = hparams.batch_size is not None

    for epoch in range(hparams.n_epochs):
        model.train()

        if use_minibatch:
            assert hparams.batch_size is not None
            batches = _make_batches(Xt, yt, hparams.batch_size, rng)
        else:
            batches = [(Xt, yt)]

        for xb_envs, yb_envs in batches:
            env_sizes = [len(x) for x in xb_envs]
            all_x = torch.cat(xb_envs, dim=0)
            all_logits = model(all_x).squeeze(-1)

            env_logits = torch.split(all_logits, env_sizes)
            env_losses = []
            env_penalties = []
            for logits_e, y_e in zip(env_logits, yb_envs, strict=True):
                loss_e = F.binary_cross_entropy_with_logits(logits_e, y_e)
                penalty_e = irm_penalty(logits_e, y_e)
                env_losses.append(loss_e)
                env_penalties.append(penalty_e)

            erm_loss = torch.stack(env_losses).mean()
            irm_penalty_val = torch.stack(env_penalties).mean()
            env_losses_tensor = torch.stack(env_losses)
            vrex_penalty_val = ((env_losses_tensor - erm_loss) ** 2).mean()

            if epoch < hparams.anneal_iters:
                irm_weight = 0.0
                vrex_weight = 0.0
            else:
                irm_weight = hparams.irm_lambda
                vrex_weight = hparams.vrex_lambda
                if not annealing_done:
                    annealing_done = True
                    best_val_metric = float("inf")
                    best_state = None
                    patience_counter = 0

            total_loss = (
                erm_loss + irm_weight * irm_penalty_val + vrex_weight * vrex_penalty_val
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # early stopping (only after annealing)
        if has_val and annealing_done and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                all_xv = torch.cat(Xv, dim=0)
                all_logits_v = model(all_xv).squeeze(-1)
                val_env_logits = torch.split(all_logits_v, [len(x) for x in Xv])
                val_losses = [
                    F.binary_cross_entropy_with_logits(lv, yv_e).item()
                    for lv, yv_e in zip(val_env_logits, yv, strict=True)
                ]
                worst_val = max(val_losses)

            if worst_val < best_val_metric:
                best_val_metric = worst_val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 10
                if patience_counter >= hparams.patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch + 1}")
                    break

    if has_val and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


@torch.no_grad()
def _predict_proba(model: TabularMLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Return P(Y=1) for each sample."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(Xt).squeeze(-1)
    prob = torch.sigmoid(logits).cpu().numpy()
    return prob


# ──────────────────────────────────────────────────────────────────────────────
# grid search
# ──────────────────────────────────────────────────────────────────────────────


def _build_base_grid() -> list[dict]:
    """Build the architecture/optimiser grid (shared across methods)."""
    grid = {
        "hidden_width": [16, 32],
        "depth": [2, 3],
        "dropout": [0.0, 0.1],
        "lr": [1e-3, 5e-4],
        "weight_decay": [0.0, 1e-4],
        "anneal_iters": [50, 100],
        "batch_size": [None],
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
        hidden_width=cfg["hidden_width"],
        depth=cfg["depth"],
        dropout=cfg["dropout"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        irm_lambda=cfg.get("irm_lambda", 0.0),
        vrex_lambda=cfg.get("vrex_lambda", 0.0),
        anneal_iters=cfg["anneal_iters"],
        batch_size=cfg.get("batch_size"),
        seed=cfg["seed"],
        n_epochs=500,
        patience=100,
    )

    model = _train_model(
        X_tr, y_tr, hp, X_val_envs=X_va, y_val_envs=y_va, device=device
    )

    val_bces = []
    for x_v, y_v in zip(X_va, y_va, strict=True):
        prob_v = _predict_proba(model, x_v, device)
        bce_v = log_loss(y_v, np.clip(prob_v, 1e-7, 1 - 1e-7), labels=[0, 1])
        val_bces.append(bce_v)

    return cfg, float(max(val_bces)), float(np.mean(val_bces))


def _grid_search(
    X_envs: list[np.ndarray],
    y_envs: list[np.ndarray],
    penalty_cfgs: list[dict],
    val_frac: float = 0.2,
    val_seed: int = 42,
    device: str = "cpu",
    n_jobs: int = 1,
    verbose: bool = False,
) -> tuple[dict | None, list[tuple[dict, float, float]]]:
    """Grid search for a specific method (IRM, V-REx, or ERM)."""
    base_grid = _build_base_grid()

    # Combine base grid with penalty configs
    grid = []
    for base_cfg in base_grid:
        for penalty_cfg in penalty_cfgs:
            grid.append({**base_cfg, **penalty_cfg})

    X_tr, y_tr, X_va, y_va = _split_train_val(
        X_envs, y_envs, val_frac=val_frac, seed=val_seed
    )

    if verbose:
        print(f"    Grid search: {len(grid)} configs, n_jobs={n_jobs}")

    raw_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_evaluate_single_config)(cfg, X_tr, y_tr, X_va, y_va, device)
        for cfg in grid
    )
    results: list[tuple[dict, float, float]] = [r for r in raw_results if r is not None]  # type: ignore[misc]

    # find best config
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
# sklearn-style Classifier Wrappers
# ──────────────────────────────────────────────────────────────────────────────


class _BaseNNClassifier(ClassifierMixin, BaseEstimator):
    """Base class for IRM/V-REx/ERM classifiers (sklearn interface)."""

    # Subclasses should set these
    _penalty_configs: list[dict] = []
    _method_name: str = "NN"

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
    ) -> "_BaseNNClassifier":
        """
        Fit the model with hyperparameter grid search.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1).
        environment : array-like of shape (n_samples,)
            Environment labels for each sample.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        environment = np.asarray(environment)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        # Standardise
        X_s, _, mu, sigma = _standardise(X)
        self._mu = mu
        self._sigma = sigma

        # Split by environment
        X_envs, y_envs, _ = _split_by_env(X_s, y, environment)

        if self.verbose:
            print(f"  Fitting {self._method_name}...")

        # Grid search
        best_cfg, self._grid_results = _grid_search(
            X_envs,
            y_envs,
            penalty_cfgs=self._penalty_configs,
            val_frac=0.2,
            val_seed=self.random_state,
            device=self.device,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        if best_cfg is None:
            raise RuntimeError("Grid search failed to find any valid configuration.")

        self.best_config_ = best_cfg

        # Retrain on full data with best config
        X_tr, y_tr, X_va, y_va = _split_train_val(
            X_envs, y_envs, val_frac=0.2, seed=self.random_state
        )

        hp = _Hparams(
            hidden_width=best_cfg["hidden_width"],
            depth=best_cfg["depth"],
            dropout=best_cfg["dropout"],
            lr=best_cfg["lr"],
            weight_decay=best_cfg["weight_decay"],
            irm_lambda=best_cfg.get("irm_lambda", 0.0),
            vrex_lambda=best_cfg.get("vrex_lambda", 0.0),
            anneal_iters=best_cfg["anneal_iters"],
            batch_size=best_cfg.get("batch_size"),
            seed=best_cfg["seed"],
            n_epochs=500,
            patience=100,
        )

        self._model = _train_model(
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
        """Return probability estimates for samples.

        Returns array of shape (n_samples, 2) with columns [P(Y=0), P(Y=1)].
        """
        X = np.asarray(X, dtype=np.float64)
        X_s = (X - self._mu) / self._sigma
        prob_1 = _predict_proba(self._model, X_s, self.device)
        return np.column_stack([1 - prob_1, prob_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class IRMClassifier(_BaseNNClassifier):
    """
    IRM (Invariant Risk Minimization) classifier.

    Uses grid search over architecture and IRM penalty weight.

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

    _penalty_configs = [
        {"irm_lambda": 1.0, "vrex_lambda": 0.0},
        {"irm_lambda": 10.0, "vrex_lambda": 0.0},
        {"irm_lambda": 100.0, "vrex_lambda": 0.0},
    ]
    _method_name = "IRM"


class VRExClassifier(_BaseNNClassifier):
    """
    V-REx (Variance Risk Extrapolation) classifier.

    Uses grid search over architecture and V-REx penalty weight.

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

    _penalty_configs = [
        {"irm_lambda": 0.0, "vrex_lambda": 1.0},
        {"irm_lambda": 0.0, "vrex_lambda": 10.0},
        {"irm_lambda": 0.0, "vrex_lambda": 100.0},
    ]
    _method_name = "V-REx"


class ERMClassifier(_BaseNNClassifier):
    """
    ERM (Empirical Risk Minimization) neural network classifier.

    Standard MLP without invariance penalties, for use as a baseline.

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

    _penalty_configs = [
        {"irm_lambda": 0.0, "vrex_lambda": 0.0},
    ]
    _method_name = "ERM (NN)"
