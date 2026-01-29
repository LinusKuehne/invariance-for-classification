"""
Conditional Randomization Test for invariance testing.

Inspired by Candès et al. (2018), this test uses conditional randomization
to test whether the conditional distribution P(Y|X_S) is invariant across environments.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from ._base import InvarianceTest


class ConditionalRandomizationTest(InvarianceTest):
    """
    Conditional Randomization Test (CRT) for invariance.

    Tests the null hypothesis H_0: E[Y|X_S, E] = E[Y|X_S] (invariance) using
    conditional randomization.

    Algorithm:
    1. Compute residuals R = Y - E[Y|X_S] using cross-validation
    2. Estimate P(E|X_S) using cross-validation for resampling
    3. Compute test statistic T_obs = ANOVA F-statistic numerator on residuals
    4. Generate M resampled environments from P(E|X_S)
    5. Compute null statistics T_null for each resample
    6. P-value = (1 + sum(T_null >= T_obs)) / (1 + M)

    Parameters
    ----------
    test_classifier_type : str, default="HGBT"
        Classifier type to use for P(Y|X_S) and P(E|X_S) estimation.
        "RF" for random forest, "HGBT" for histogram gradient boosting.
        "LR" is also accepted and defaults to "HGBT".

    n_permutations : int, default=200
        Number of environment resamplings for the null distribution.

    n_folds : int, default=5
        Number of folds for cross-fitting.

    random_state : int, default=42
        Random state for reproducibility.

    References
    ----------
    Candès, E., Fan, Y., Janson, L., & Lv, J. (2018).
    Panning for Gold: Model-X Knockoffs for High-dimensional Controlled Variable Selection.
    Journal of the Royal Statistical Society: Series B, 80(3), 551-577.
    """

    def __init__(
        self,
        test_classifier_type: str = "HGBT",
        n_permutations: int = 200,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        # validate classifier type
        if test_classifier_type not in ("RF", "LR", "HGBT"):
            raise ValueError(
                f"Unknown test_classifier_type: {test_classifier_type}. "
                "Must be 'RF', 'LR', or 'HGBT'."
            )
        # LR defaults to HGBT for this test
        if test_classifier_type == "LR":
            test_classifier_type = "HGBT"

        self.test_classifier_type = test_classifier_type
        self.n_permutations = n_permutations
        self.n_folds = n_folds
        self.random_state = random_state
        self.name = "crt"

    def _get_y_model(self):
        """Get the model for P(Y | X_S) estimation."""
        if self.test_classifier_type == "RF":
            return RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=1
            )
        elif self.test_classifier_type == "HGBT":
            return HistGradientBoostingClassifier(random_state=self.random_state)
        else:
            raise ValueError(
                f"Unknown test_classifier_type: {self.test_classifier_type}"
            )

    def _get_env_model(self):
        """Get the model for P(E | X_S) estimation."""
        if self.test_classifier_type == "RF":
            return RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=1
            )
        elif self.test_classifier_type == "HGBT":
            return HistGradientBoostingClassifier(random_state=self.random_state)
        else:
            raise ValueError(
                f"Unknown test_classifier_type: {self.test_classifier_type}"
            )

    def _calc_statistic(self, residuals: np.ndarray, envs: np.ndarray) -> float:
        """
        Calculate ANOVA-like test statistic for residual-environment dependence.

        T = sum_e n_e * (mean(residuals_e) - global_mean)^2

        This detects if the residual mean shifts across environments.
        """
        unique_envs = np.unique(envs)
        global_mean = np.mean(residuals)
        stat = 0.0

        for e in unique_envs:
            idx = envs == e
            n_e = np.sum(idx)
            if n_e > 0:
                diff = np.mean(residuals[idx]) - global_mean
                stat += n_e * (diff**2)

        return stat

    def test(self, X: np.ndarray, y: np.ndarray, E: np.ndarray) -> float:
        """
        Perform the conditional randomization test.

        Parameters
        ----------
        X : np.ndarray
            Predictor subset of shape (n_samples, n_features).
        y : np.ndarray
            Target variable of shape (n_samples,), assumed binary {0, 1}.
        E : np.ndarray
            Environment labels of shape (n_samples,).

        Returns
        -------
        p_value : float
            P-value of the test. High values indicate invariance.
        """
        # Handle single environment case
        if len(np.unique(E)) < 2:
            return 1.0

        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        unique_envs = np.unique(E)
        n_envs = len(unique_envs)

        # Step 1: Compute residuals R = Y - E[Y|X_S]
        if X.shape[1] == 0:
            # Empty feature set: E[Y|X_S] = E[Y] (global mean)
            y_pred = np.full(n_samples, np.mean(y))
        else:
            model_y = self._get_y_model()
            try:
                cv = StratifiedKFold(
                    n_splits=self.n_folds, shuffle=True, random_state=self.random_state
                )
                y_pred = cross_val_predict(
                    model_y, X, y, cv=cv, method="predict_proba", n_jobs=1
                )[:, 1]
            except ValueError:
                cv = KFold(
                    n_splits=self.n_folds, shuffle=True, random_state=self.random_state
                )
                y_pred = cross_val_predict(
                    model_y, X, y, cv=cv, method="predict_proba", n_jobs=1
                )[:, 1]

        residuals = y - y_pred

        # Step 2: Estimate P(E | X_S) for resampling
        if X.shape[1] == 0:
            # Empty feature set: P(E|X_S) = P(E) (marginal distribution)
            probs_E = np.zeros((n_samples, n_envs))
            for i, e in enumerate(unique_envs):
                probs_E[:, i] = np.mean(E == e)
            classes_E = unique_envs
        else:
            model_e = self._get_env_model()
            try:
                cv = StratifiedKFold(
                    n_splits=self.n_folds, shuffle=True, random_state=self.random_state
                )
                probs_E = cross_val_predict(
                    model_e, X, E, cv=cv, method="predict_proba", n_jobs=1
                )
            except ValueError:
                cv = KFold(
                    n_splits=self.n_folds, shuffle=True, random_state=self.random_state
                )
                probs_E = cross_val_predict(
                    model_e, X, E, cv=cv, method="predict_proba", n_jobs=1
                )
            # Get classes from a fitted model
            model_e.fit(X, E)
            classes_E = model_e.classes_

        # Step 3: Compute observed test statistic
        t_obs = self._calc_statistic(residuals, E)

        # Step 4: Generate null distribution via environment resampling
        t_nulls = np.zeros(self.n_permutations)

        for i in range(self.n_permutations):
            # Sample E_tilde from P(E | X_S) for each observation
            cumsum = np.cumsum(probs_E, axis=1)
            rand_vals = rng.random((n_samples, 1))
            E_indices = (cumsum < rand_vals).sum(axis=1)
            E_indices = np.clip(E_indices, 0, n_envs - 1)
            E_tilde = classes_E[E_indices]

            t_nulls[i] = self._calc_statistic(residuals, E_tilde)

        # Step 5: Compute p-value
        p_value = (1 + np.sum(t_nulls >= t_obs)) / (1 + self.n_permutations)

        return float(p_value)
