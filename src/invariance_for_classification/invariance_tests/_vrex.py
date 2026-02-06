"""
V-REx-inspired Invariance Test.

This implements an invariance test inspired by the Variance Risk Extrapolation (V-REx)
concept. The test compares the "regret" of using a global predictor (trained on
subset X_S across all environments) versus environment-specific predictors (trained
on X_S within each environment separately).

The intuition: if Y|X_S is truly invariant across environments, then environment-
specific models should not systematically outperform the global model in any
particular environment, i.e., the regret distribution should be similar across
environments.

Inspired by the V-REx penalty idea from:
Krueger et al. "Out-of-Distribution Generalization via Risk Extrapolation"
https://arxiv.org/abs/2003.00688
"""

from typing import Optional

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold

from ._base import InvarianceTest


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


class VRExTest(InvarianceTest):
    """
    V-REx-inspired Invariance Test.

    Tests the null hypothesis that Y | X_S is invariant across environments by
    comparing the "regret" distribution across environments.

    Algorithm:
    1. Train a global model f on X_S (subset of predictors) using all data
    2. For each environment e, train an environment-specific model g_e on X_S
       using only data from environment e
    3. Compute regret for each observation: D_i = BCE(y_i, f(x_S_i)) - BCE(y_i, g_e(x_S_i))
    4. Perform Kruskal-Wallis test on regrets grouped by environment

    H0: The regret distributions are the same across environments
    (which implies Y|X_S is invariant)

    Parameters
    ----------
    test_classifier_type : str or None, default=None
        Classifier type to use. Supported values:
        - None or "RF": Random Forest (uses OOB predictions)
        - "HGBT": Histogram Gradient Boosting (uses cross-validation)
        Note: "LR" is accepted for API compatibility but defaults to "RF".
    n_estimators : int, default=100
        Number of trees in the random forest (only used for RF).
    use_anova : bool, default=False
        If True, use one-way ANOVA instead of Kruskal-Wallis test.
        Kruskal-Wallis is the default as it's non-parametric.
    n_folds : int, default=5
        Number of folds for cross-validation (only used for HGBT).
    random_state : int or None, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        test_classifier_type: Optional[str] = None,
        n_estimators: int = 100,
        use_anova: bool = False,
        n_folds: int = 5,
        random_state: Optional[int] = 42,
    ):
        # Validate test_classifier_type
        valid_types = [None, "RF", "LR", "HGBT"]
        if test_classifier_type not in valid_types:
            raise ValueError(
                f"Unknown test_classifier_type: {test_classifier_type}. "
                f"Valid options are: {valid_types}"
            )

        # LR defaults to RF for this test
        if test_classifier_type == "LR":
            test_classifier_type = "RF"

        self.test_classifier_type = (
            test_classifier_type if test_classifier_type else "RF"
        )
        self.n_estimators = n_estimators
        self.use_anova = use_anova
        self.n_folds = n_folds
        self.random_state = random_state
        self.name = "vrex"

    def test(self, X: np.ndarray, y: np.ndarray, E: np.ndarray) -> float:
        """
        Perform the V-REx invariance test.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Subset of predictors to test (X_S).
        y : np.ndarray of shape (n_samples,)
            Target variable, binary {0, 1}.
        E : np.ndarray of shape (n_samples,)
            Environment indicator.

        Returns
        -------
        p_value : float
            The p-value of the test. High p-values indicate that the null hypothesis
            (invariance) cannot be rejected.
        """
        unique_envs = np.unique(E)
        n_envs = len(unique_envs)

        # single environment case: cannot test, return 1.0
        if n_envs < 2:
            return 1.0

        y = np.asarray(y).astype(int)
        E = np.asarray(E)

        # get predictions from global model f trained on X_S
        global_preds = self._get_global_predictions(X, y)

        # get predictions from environment-specific models g_e trained on X_S
        # using only data from each environment e
        env_specific_preds = self._get_env_specific_predictions(X, y, E, unique_envs)

        # compute element-wise BCE losses
        bce_global = _binary_cross_entropy(y, global_preds)
        bce_env_specific = _binary_cross_entropy(y, env_specific_preds)

        regrets = bce_global - bce_env_specific

        regret_groups = [regrets[E == e] for e in unique_envs]

        # check if we have enough data in each group
        if any(len(g) < 2 for g in regret_groups):
            return 1.0

        # statistical test on regret distributions
        try:
            if self.use_anova:
                # one-way ANOVA (assumes normality)
                _, p_value = stats.f_oneway(*regret_groups)
            else:
                # kruskal-Wallis test (non-parametric)
                _, p_value = stats.kruskal(*regret_groups)
        except ValueError:
            return 1.0

        if np.isnan(p_value):
            return 1.0

        return float(p_value)

    def _get_global_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get out-of-sample predictions from global model trained on all data.
        Uses OOB predictions for RF, cross-validation for HGBT.
        """
        n = len(y)

        # handle empty feature set
        if X.shape[1] == 0:
            mean_y = np.mean(y)
            return np.full(n, mean_y)

        if self.test_classifier_type == "HGBT":
            return self._get_cv_predictions_hgbt(X, y)

        # Default: RF with OOB
        estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=1,
        )
        estimator.fit(X, y)

        preds = estimator.oob_decision_function_[:, 1]
        # handle any NaN OOB predictions
        if np.any(np.isnan(preds)):
            fallback = estimator.predict_proba(X)[:, 1]
            preds = np.where(np.isnan(preds), fallback, preds)
        return preds

    def _get_cv_predictions_hgbt(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get cross-validated predictions using HGBT."""
        n = len(y)
        preds = np.zeros(n)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in kf.split(X):
            estimator = HistGradientBoostingClassifier(random_state=self.random_state)
            estimator.fit(X[train_idx], y[train_idx])
            preds[test_idx] = estimator.predict_proba(X[test_idx])[:, 1]

        return preds

    def _get_env_specific_predictions(
        self, X: np.ndarray, y: np.ndarray, E: np.ndarray, unique_envs: np.ndarray
    ) -> np.ndarray:
        """
        Get out-of-sample predictions from environment-specific models.

        For each environment e, train a model only on data from that
        environment and get out-of-sample predictions.
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

            if self.test_classifier_type == "HGBT":
                # Use cross-validation within environment for HGBT
                env_preds = self._get_cv_predictions_hgbt(X_e, y_e)
            else:
                # RF with OOB
                estimator = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=1,
                )
                estimator.fit(X_e, y_e)

                env_preds = estimator.oob_decision_function_[:, 1]
                if np.any(np.isnan(env_preds)):
                    fallback = estimator.predict_proba(X_e)[:, 1]
                    env_preds = np.where(np.isnan(env_preds), fallback, env_preds)

            preds[mask] = env_preds

        return preds
