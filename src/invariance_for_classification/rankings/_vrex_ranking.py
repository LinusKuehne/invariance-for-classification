"""
V-REx-inspired Invariance Ranking.

This implements a ranking metric inspired by the Variance Risk Extrapolation (V-REx)
concept. Unlike the VRExTest which computes a p-value, this ranking directly uses
the negative variance of mean regrets across environments.

The intuition: if Y|X_S is truly invariant across environments, then the mean regret
(difference between global and environment-specific model losses) should be similar
across environments, resulting in low variance.

We return the negative variance so that higher values indicate "more invariant"
(consistent with wanting to maximize the ranking score for invariant subsets).

Inspired by the V-REx penalty idea from:
Krueger et al. "Out-of-Distribution Generalization via Risk Extrapolation"
https://arxiv.org/abs/2003.00688
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..invariance_tests._vrex import _binary_cross_entropy


def _get_global_predictions(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """
    Get out-of-sample predictions from global model trained on all data.
    Uses OOB predictions from Random Forest.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target variable, binary {0, 1}.
    n_estimators : int, default=100
        Number of trees in the random forest.
    random_state : int or None, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Out-of-sample predicted probabilities for class 1.
    """
    n = len(y)

    # handle empty feature set
    if X.shape[1] == 0:
        mean_y = np.mean(y)
        return np.full(n, mean_y)

    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        oob_score=True,
        random_state=random_state,
        n_jobs=1,
    )
    estimator.fit(X, y)

    preds = estimator.oob_decision_function_[:, 1]
    # handle any NaN OOB predictions
    if np.any(np.isnan(preds)):
        fallback = estimator.predict_proba(X)[:, 1]
        preds = np.where(np.isnan(preds), fallback, preds)
    return preds


def _get_env_specific_predictions(
    X: np.ndarray,
    y: np.ndarray,
    E: np.ndarray,
    unique_envs: np.ndarray,
    n_estimators: int = 100,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """
    Get out-of-sample predictions from environment-specific models.

    For each environment e, train a Random Forest only on data from that
    environment and get OOB predictions for observations in that environment.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target variable, binary {0, 1}.
    E : np.ndarray of shape (n_samples,)
        Environment indicator.
    unique_envs : np.ndarray
        Array of unique environment values.
    n_estimators : int, default=100
        Number of trees in the random forest.
    random_state : int or None, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Out-of-sample predicted probabilities for class 1.
    """
    n = len(y)
    preds = np.zeros(n)

    # handle empty feature set
    if X.shape[1] == 0:
        for e in unique_envs:
            mask = E == e
            preds[mask] = np.mean(y[mask])
        return preds

    for e in unique_envs:
        mask = E == e
        X_e = X[mask]
        y_e = y[mask]
        n_e = len(y_e)

        # need at least some samples to train
        if n_e < 5:
            preds[mask] = np.mean(y_e) if n_e > 0 else 0.5
            continue

        # check if y_e has both classes
        if len(np.unique(y_e)) < 2:
            preds[mask] = np.mean(y_e)
            continue

        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            oob_score=True,
            random_state=random_state,
            n_jobs=1,
        )
        estimator.fit(X_e, y_e)

        env_preds = estimator.oob_decision_function_[:, 1]
        if np.any(np.isnan(env_preds)):
            fallback = estimator.predict_proba(X_e)[:, 1]
            env_preds = np.where(np.isnan(env_preds), fallback, env_preds)
        preds[mask] = env_preds

    return preds


def vrex_ranking(
    Y: np.ndarray,
    E: np.ndarray,
    X_S: np.ndarray,
    n_estimators: int = 100,
    random_state: Optional[int] = 42,
) -> float:
    """
    Compute the V-REx-inspired invariance ranking score.

    This function computes the negative variance of mean regrets across environments.
    Higher values (less negative) indicate more invariance, as similar mean regrets
    across environments suggest that Y|X_S is invariant.

    Algorithm:
    1. Train a global model f on X_S using all data
    2. For each environment e, train an environment-specific model g_e on X_S
    3. Compute regret for each observation: D_i = BCE(y_i, f(x_S_i)) - BCE(y_i, g_e(x_S_i))
    4. Compute mean regret within each environment
    5. Return negative variance of these |E| mean regrets

    Parameters
    ----------
    Y : np.ndarray of shape (n_samples,)
        Target variable, binary {0, 1}.
    E : np.ndarray of shape (n_samples,)
        Environment indicator.
    X_S : np.ndarray of shape (n_samples, n_features)
        Subset of predictors to evaluate.
    n_estimators : int, default=100
        Number of trees in the random forest.
    random_state : int or None, default=42
        Random seed for reproducibility.

    Returns
    -------
    float
        Negative variance of mean regrets across environments.
        Higher values (closer to 0) indicate more invariance.
        Returns 0.0 for edge cases (single environment, insufficient data).
    """
    unique_envs = np.unique(E)
    n_envs = len(unique_envs)

    # single environment case: cannot compute variance, return 0.0
    if n_envs < 2:
        return 0.0

    Y = np.asarray(Y).astype(int)
    E = np.asarray(E)

    # get predictions from global model f trained on X_S
    global_preds = _get_global_predictions(X_S, Y, n_estimators, random_state)

    # get predictions from environment-specific models g_e trained on X_S
    env_specific_preds = _get_env_specific_predictions(
        X_S, Y, E, unique_envs, n_estimators, random_state
    )

    # compute element-wise BCE losses
    bce_global = _binary_cross_entropy(Y, global_preds)
    bce_env_specific = _binary_cross_entropy(Y, env_specific_preds)

    # compute regrets: positive regret means global model is worse than env-specific
    regrets = bce_global - bce_env_specific

    # compute mean regret within each environment
    mean_regrets = np.array([np.mean(regrets[E == e]) for e in unique_envs])

    # check for NaN values (can happen with edge cases)
    if np.any(np.isnan(mean_regrets)):
        return 0.0

    # return negative variance of mean regrets
    # higher values (less negative) indicate more invariance
    variance = float(np.var(mean_regrets))

    return -variance
