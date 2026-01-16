from typing import Any, Optional

import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.ensemble import RandomForestClassifier

from ._base import InvarianceTest


class InvariantResidualDistributionTest(InvarianceTest):
    """
    invariant residual distribution test

    Tests the null hypothesis that the residuals of a model predicting Y from X_S
    have the same mean across all envs E.

    H0: mean(residuals | E=e) constant for all e

    Algorithm:
    1. Fit a model (e.g., RF) to predict Y given X_S
    2. Compute residuals R = Y - P(Y=1|X_S)
    3. Perform one-way ANOVA on R grouped by E

    For RFs, OOB predictions are preferred to avoid overfitting residuals
    """

    def __init__(self, estimator: Optional[BaseEstimator] = None):
        """
        Parameters
        ----------
        estimator : BaseEstimator, optional
            base estimator to use for predicting Y from X_S
            Defaults to RandomForestClassifier(n_estimators=100, oob_score=True)
        """
        if estimator is None:
            self.estimator = RandomForestClassifier(
                n_estimators=100, oob_score=True, random_state=42
            )
        else:
            self.estimator = estimator

    def test(self, X: np.ndarray, y: np.ndarray, E: np.ndarray) -> float:
        """
        Perform invariant residual distr test.

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
        unique_envs = np.unique(E)
        if len(unique_envs) < 2:
            # test undefined for single environment; assume invariance (cannot reject)
            return 1.0

        # case 1: empty set
        # => checking if P(Y) changes across envs
        if X.shape[1] == 0:
            mean_y = np.mean(y)
            y_pred_proba = np.full_like(y, mean_y, dtype=float)

        # case 2: non-empty set of predictors
        else:
            est = clone(self.estimator)

            # enable OOB if available
            use_oob = False
            if hasattr(est, "oob_score"):
                est.set_params(oob_score=True)
                use_oob = True

            est.fit(X, y)

            # retrieve predictions (OOB preferred)
            y_pred_proba = self._get_predictions(est, X, use_oob)

        residuals = y - y_pred_proba

        groups = [residuals[E == e] for e in unique_envs]

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
            if is_classifier(est):
                classes = est.classes_
                if 1 in classes:
                    col_idx = np.where(classes == 1)[0][0]
                    oob = est.oob_decision_function_[:, col_idx]
                    if np.any(np.isnan(oob)):
                        fallback = est.predict_proba(X)[:, col_idx]
                        oob = np.where(np.isnan(oob), fallback, oob)
                    return oob
                return np.zeros(X.shape[0])
            return est.oob_prediction_

        if is_classifier(est):
            classes = est.classes_
            if 1 in classes:
                col_idx = np.where(classes == 1)[0][0]
                return est.predict_proba(X)[:, col_idx]
            return np.zeros(X.shape[0])
        return est.predict(X)
