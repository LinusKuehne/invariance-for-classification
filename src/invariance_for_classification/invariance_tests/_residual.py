from typing import Any

import numpy as np
import scipy.stats as stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ._base import InvarianceTest


class InvariantResidualDistributionTest(InvarianceTest):
    """
    Invariant residual distribution test

    Tests the null hypothesis that the residuals of a model predicting Y from X_S
    have the same mean across all envs E.

    H_0: mean(residuals | E=e) constant for all e

    Algorithm:
    1. Fit a model (e.g., RF) to predict Y given X_S
    2. Compute residuals R = Y - P(Y=1|X_S)
    3. Perform one-way ANOVA on R grouped by E

    For RFs, OOB predictions are preferred to avoid overfitted residuals.
    For logistic regression, we use cross-validation.

    Parameters
    ----------
    test_classifier_type : str, default="RF"
        "RF" for random forest, "LR" for logistic regression.
    """

    def __init__(
        self,
        test_classifier_type: str = "RF",
    ):
        self.test_classifier_type = test_classifier_type
        if test_classifier_type == "RF":
            self.estimator = RandomForestClassifier(
                n_estimators=100, oob_score=True, random_state=42, n_jobs=1
            )
        elif test_classifier_type == "LR":
            self.estimator = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unknown test_classifier_type: {test_classifier_type}")

        self.name = "inv_residual"

    def test(self, X: np.ndarray, y: np.ndarray, E: np.ndarray) -> float:
        """
        Perform the invariant residual distribution test.

        Parameters
        ----------
        X : np.ndarray
             predictor subset of shape (n_samples, n_features)
        y : np.ndarray
             target variable of shape (n_samples,), assumed binary {0, 1}
        E : np.ndarray
             environment labels of shape (n_samples,)

        Returns
        -------
        p_value : float
            p-value of one-way ANOVA test
        """
        if len(np.unique(E)) < 2:
            # test undefined for single environment; assume invariance (cannot reject)
            return 1.0

        # case 1: empty set
        # => checking if P(Y) changes across envs
        if X.shape[1] == 0:
            mean_y = np.mean(y)
            y_pred = np.full_like(y, mean_y, dtype=float)

        # case 2: non-empty set of predictors
        else:
            est = clone(self.estimator)

            # enable OOB if available
            use_oob = False
            if hasattr(est, "oob_score"):
                est.set_params(oob_score=True)
                use_oob = True

            if use_oob:
                est.fit(X, y)
                y_pred = self._get_predictions(est, X, use_oob)
            else:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                y_pred = cross_val_predict(
                    est, X, y, cv=cv, method="predict_proba", n_jobs=1
                )[:, 1]

        residuals = y - y_pred

        groups = [residuals[E == e] for e in np.unique(E)]

        # stats.f_oneway handles variance checks internally to some degree
        try:
            _, p_value = stats.f_oneway(*groups)
        except ValueError:
            # can happen if all values are identical in groups
            return 1.0

        if np.isnan(p_value):
            return 1.0

        return p_value

    def _get_predictions(self, est: Any, X: np.ndarray, use_oob: bool) -> np.ndarray:
        """Helper to get probability predictions (using OOB if possible)."""
        if use_oob and hasattr(est, "oob_decision_function_"):
            oob = est.oob_decision_function_[:, 1]
            if np.any(np.isnan(oob)):
                fallback = est.predict_proba(X)[:, 1]
                oob = np.where(np.isnan(oob), fallback, oob)
            return oob

        return est.predict_proba(X)[:, 1]
