"""MaxRM Random Forest classifier for binary classification (posthoc).

Wraps the adaXT regression forest with the MaxRM posthoc optimization
(``modify_predictions_trees``) to minimize the worst-case Brier score
across environments.  Since Brier score = MSE for probability estimates
vs. binary labels, the existing regression-based MaxRM machinery applies
directly.

Requires the ``adaXT`` fork from https://github.com/francescofreni/adaXT
and the ``nldg`` package from https://github.com/francescofreni/nldg.
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MaxRMRFClassifier(BaseEstimator, ClassifierMixin):
    """MaxRM Random Forest for binary classification (posthoc Brier-score variant).

    Fits an adaXT *regression* forest on 0/1 labels, then applies
    ``modify_predictions_trees(E, method="mse")`` to adjust leaf predictions
    so that the worst-case Brier score across environments is minimised.

    The raw regression outputs are clipped to [0, 1] and returned as class
    probabilities.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    random_state : int or None, default=None
        Seed for the random number generator.
    min_samples_leaf : int, default=5
        Minimum number of samples in a leaf.  Values > 1 recommended so that
        leaves contain observations from multiple environments.
    max_features : int, float, str or None, default="sqrt"
        Number of features to consider at each split (passed to adaXT).
    maxrm_solver : str or None, default=None
        Solver for the CVXPY problem inside ``modify_predictions_trees``.
        If ``None``, tries CLARABEL → ECOS → SCS automatically.
    n_jobs_maxrm : int, default=1
        Parallelism for the posthoc optimisation step.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        min_samples_leaf: int = 5,
        max_features: int | float | Literal["sqrt", "log2"] | None = "sqrt",
        maxrm_solver: Optional[str] = None,
        n_jobs_maxrm: int = 1,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.maxrm_solver = maxrm_solver
        self.n_jobs_maxrm = n_jobs_maxrm

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        environment: Optional[np.ndarray] = None,
    ) -> "MaxRMRFClassifier":
        """Fit the MaxRM-RF classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Binary labels (0/1).
        environment : array-like of shape (n_samples,) or None
            Environment labels.  When provided the MaxRM posthoc step
            is applied; otherwise the model is a plain regression forest.

        Returns
        -------
        self
        """
        from adaXT.random_forest import RandomForest

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.classes_ = np.unique(y.astype(int))

        seed = (
            self.random_state
            if isinstance(self.random_state, (int, np.integer))
            else None
        )

        # adaXT expects Literal["sqrt","log2"] | int | float | None;
        # cast to satisfy both adaXT and the type checker.
        max_feat: int | float | None = self.max_features  # type: ignore[assignment]

        self.rf_ = RandomForest(
            forest_type="Regression",
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=max_feat,
            seed=seed,
        )
        self.rf_.fit(X, y)

        # Apply MaxRM posthoc step to minimise worst-case Brier score
        if environment is not None:
            self._apply_maxrm(np.asarray(environment))

        return self

    # ------------------------------------------------------------------

    def _apply_maxrm(self, environment: np.ndarray) -> None:
        """Run ``modify_predictions_trees`` with fallback solvers."""
        candidate_solvers = []
        if self.maxrm_solver is not None:
            candidate_solvers.append(self.maxrm_solver)
        for s in ["CLARABEL", "ECOS", "SCS"]:
            if s not in candidate_solvers:
                candidate_solvers.append(s)

        success = False
        for solver in candidate_solvers:
            try:
                self.rf_.modify_predictions_trees(
                    E=environment,
                    method="mse",  # Brier score = MSE for 0/1 targets
                    solver=solver,
                    n_jobs=self.n_jobs_maxrm,
                )
                success = True
                break
            except Exception:
                continue

        if not success:
            try:
                self.rf_.modify_predictions_trees(
                    E=environment,
                    method="mse",
                    opt_method="extragradient",
                    n_jobs=self.n_jobs_maxrm,
                )
            except Exception:
                warnings.warn(
                    "MaxRM posthoc optimisation failed for all solvers. "
                    "Falling back to standard RF predictions.",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape (n_samples, 2)."""
        X = np.asarray(X, dtype=np.float64)
        raw = self.rf_.predict(X).ravel()
        prob_pos = np.clip(raw, 0.0, 1.0)
        return np.column_stack([1.0 - prob_pos, prob_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]
