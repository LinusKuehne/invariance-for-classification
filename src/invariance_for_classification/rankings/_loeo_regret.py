"""
LOEO (Leave-One-Environment-Out) Regret Ranking.

This implements a ranking metric using a Leave-One-Environment-Out (LOEO) regret approach.

For each held-out environment e:
  1. Train a global model on all other environments → evaluate on e → global_loss_e
  2. Train individual models on each other env e' → evaluate on e → env_loss_e'
  3. regret_e = global_loss_e - mean(env_loss_e')

Returns a dictionary with aggregate scores (mean and min of regrets).
Higher values (closer to 0) indicate "more invariant".

Intuition: If Y|X_S is truly invariant, a model pooling data from multiple
environments should perform comparably to individual environment models when
transferred to a new environment. The regret should be consistently small
across held-out environments.

The regret formulation normalises away base-rate differences across environments,
making the ranking robust to datasets where P(Y=1) varies across environments.
"""

from typing import Literal, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def _binary_cross_entropy(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15
) -> np.ndarray:
    """
    Compute element-wise binary cross-entropy loss.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted probabilities for class 1.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    np.ndarray
        Element-wise BCE loss values.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def _make_classifier(
    classifier_type: Literal["RF", "HGBT"] = "HGBT",
    random_state: Optional[int] = 42,
    n_estimators: int = 100,
) -> RandomForestClassifier | HistGradientBoostingClassifier:
    """
    Create a classifier with appropriate regularisation.

    Parameters
    ----------
    classifier_type : {"RF", "HGBT"}, default="HGBT"
        Classifier type to create.
    random_state : int or None, default=42
        Random seed for reproducibility.
    n_estimators : int, default=100
        Number of trees (RF only).

    Returns
    -------
    RandomForestClassifier or HistGradientBoostingClassifier
    """
    if classifier_type == "HGBT":
        return HistGradientBoostingClassifier(
            random_state=random_state,
            min_samples_leaf=20,
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
        )
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,
    )


def _mean_bce_loss(
    clf: RandomForestClassifier | HistGradientBoostingClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """
    Train *clf* on (X_train, y_train) and return mean BCE loss on (X_test, y_test).

    If the training set contains only one class, the marginal rate is predicted
    instead of fitting the classifier.
    """
    if len(np.unique(y_train)) < 2:
        p = float(np.mean(y_train))
        return float(np.mean(_binary_cross_entropy(y_test, np.full(len(y_test), p))))

    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:, 1]
    return float(np.mean(_binary_cross_entropy(y_test, preds)))


def loeo_regret(
    Y: np.ndarray,
    E: np.ndarray,
    X_S: np.ndarray,
    n_estimators: int = 100,
    random_state: Optional[int] = 42,
    classifier_type: Literal["RF", "HGBT"] = "HGBT",
) -> dict[str, float]:
    """
    Compute LOEO (Leave-One-Environment-Out) regret scores.

    Uses a Leave-One-Environment-Out scheme:

    For each held-out environment *e*:
      1. Train a **global** model on data from all other environments and
         compute its mean BCE loss on *e* → ``global_loss_e``.
      2. For every other environment *e'*, train an **env-specific** model on
         *e'* alone and compute its mean BCE loss on *e* → ``env_loss_{e',e}``.
      3. ``regret_e = global_loss_e - mean(env_loss_{e',e}  for e' ≠ e)``

    Returns a dictionary containing aggregated scores based on the regrets:
      - 'mean': Mean of regrets.
      - 'min': Minimum of regrets.

    Higher values (closer to 0) indicate more invariance. A very negative regret
    means the global model performed much worse than environment-specific models
    (indicating environment-specificity).

    Parameters
    ----------
    Y : np.ndarray of shape (n_samples,)
        Target variable, binary {0, 1}.
    E : np.ndarray of shape (n_samples,)
        Environment indicator.
    X_S : np.ndarray of shape (n_samples, n_features)
        Subset of predictors to evaluate.
    n_estimators : int, default=100
        Number of trees in the random forest (only used for RF).
    random_state : int or None, default=42
        Random seed for reproducibility.
    classifier_type : {"RF", "HGBT"}, default="HGBT"
        Classifier type to use. HGBT is recommended for best performance.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'mean' and 'min'.
        Returns {'mean': 0.0, 'min': 0.0} for edge cases (single environment).
    """
    unique_envs = np.unique(E)
    n_envs = len(unique_envs)

    # single environment case: cannot compute regrets, return 0.0
    if n_envs < 2:
        return {"mean": 0.0, "min": 0.0}

    Y = np.asarray(Y).astype(int)
    E = np.asarray(E)

    regrets: list[float] = []

    for e in unique_envs:
        mask_test = E == e
        mask_train = E != e
        other_envs = [e2 for e2 in unique_envs if e2 != e]
        y_test = Y[mask_test]

        # --- global model trained on all other environments ---
        if X_S.shape[1] == 0:
            p = float(np.mean(Y[mask_train]))
            global_loss = float(
                np.mean(_binary_cross_entropy(y_test, np.full(len(y_test), p)))
            )
        else:
            clf = _make_classifier(classifier_type, random_state, n_estimators)
            global_loss = _mean_bce_loss(
                clf, X_S[mask_train], Y[mask_train], X_S[mask_test], y_test
            )

        # --- individual env-specific models (one per other environment) ---
        env_losses: list[float] = []
        for e2 in other_envs:
            mask_e2 = E == e2
            if X_S.shape[1] == 0:
                p = float(np.mean(Y[mask_e2]))
                eloss = float(
                    np.mean(_binary_cross_entropy(y_test, np.full(len(y_test), p)))
                )
            else:
                clf = _make_classifier(classifier_type, random_state, n_estimators)
                eloss = _mean_bce_loss(
                    clf, X_S[mask_e2], Y[mask_e2], X_S[mask_test], y_test
                )
            env_losses.append(eloss)

        avg_env_loss = float(np.mean(env_losses))
        regrets.append(global_loss - avg_env_loss)

    regrets_arr = np.array(regrets)

    # check for NaN values (can happen with edge cases)
    if np.any(np.isnan(regrets_arr)):
        return {"mean": 0.0, "min": 0.0}

    return {
        "mean": float(np.mean(regrets_arr)),
        "min": float(np.min(regrets_arr)),
    }
