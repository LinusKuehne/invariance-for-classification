import logging
from itertools import chain, combinations
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state, resample
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..invariance_tests import InvarianceTest, InvariantResidualDistributionTest

logger = logging.getLogger(__name__)


class _EmptySetClassifier(BaseEstimator, ClassifierMixin):
    """
    Internal dummy classifier for the empty feature set
    Always predicts the class prior distribution observed during fit
    """

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            self.prior_ = np.array([1.0])
        else:
            self.prior_ = np.array([np.mean(y == c) for c in self.classes_])
        # mock OOB decision function (optimistic, just repeats prior)
        n_samples = len(y)
        self.oob_decision_function_ = np.tile(self.prior_, (n_samples, 1))
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        return np.tile(self.prior_, (n_samples, 1))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class StabilizedClassificationClassifier(BaseEstimator, ClassifierMixin):
    """
    Stabilized classification classifier

    Ensemble method that averages predictions from multiple classifiers
    trained on subsets of features that are "invariant" and "predictive"

    Algorithm:
    1. iterate over all feature subsets
    2. test each subset for invariance using `invariance_test`
    3. for invariant subsets, measure predictive performance (binary cross-entropy)
    4. determine a predictive cutoff using bootstrapping on the single best invariant subset
    5. form an ensemble of all invariant subsets exceeding this predictive cutoff

    Parameters
    ----------
    alpha_inv : float, default=0.05
        significance level for the invariance test. Subsets with p-value >= alpha_inv
        are considered invariant

    alpha_pred : float, default=0.05
        parameter controlling the predictive score cutoff (related to the quantile
        of the bootstrap distribution of the best model's performance)

    estimator : BaseEstimator, optional
        the base classifier to use for each subset
        Defaults to RandomForestClassifier(n_estimators=100, oob_score=True)

    invariance_test : InvarianceTest, optional
        the invariance test to use
        Defaults to InvariantResidualDistributionTest

    n_bootstrap : int, default=100
        number of bootstrap samples used to determine the predictive cutoff

    verbose : int, default=0
        verbosity level

    random_state : int, RandomState instance or None, default=None
        random state of the estimator
    """

    def __init__(
        self,
        alpha_inv: float = 0.05,
        alpha_pred: float = 0.05,
        estimator: Optional[BaseEstimator] = None,
        invariance_test: Optional[InvarianceTest] = None,
        n_bootstrap: int = 100,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ):
        self.alpha_inv = alpha_inv
        self.alpha_pred = alpha_pred
        self.estimator = estimator
        self.invariance_test = invariance_test
        self.n_bootstrap = n_bootstrap
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None, environment=None):
        """
        Fit the model

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data
        y : array-like or str, optional
            Target values (binary) or column name in X
        environment : array-like or str
            environment labels or column name in X

        Returns
        -------
        self : object
        """
        X, y, environment = self._validate_input(X, y, environment)

        self.random_state_ = check_random_state(self.random_state)

        # Encoder target y to 0..K-1 integers
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        if len(self.classes_) > 2:
            raise ValueError(
                f"Only binary classification is supported. Found classes: {self.classes_}"
            )

        n_features = X.shape[1]
        self.n_features_in_ = n_features

        # resolve dependencies
        base_estimator = self.estimator or RandomForestClassifier(
            n_estimators=100, oob_score=True, random_state=self.random_state
        )
        inv_test = self.invariance_test or InvariantResidualDistributionTest(
            estimator=base_estimator
        )

        # 1. & 2.: determine invariant subsets
        subset_stats, all_p_values = self._find_invariant_subsets(
            X, y_encoded, environment, n_features, inv_test, base_estimator
        )

        # fallback if no invariant subsets found
        if not subset_stats:
            if self.verbose:
                logger.warning(
                    "No invariant subsets found. Using subset with max p-value."
                )
            subset_stats = self._apply_fallback(
                X, y_encoded, all_p_values, base_estimator
            )

        # 3. compute predictive cutoff via bootstrap
        cutoff = self._compute_cutoff(X, y_encoded, subset_stats, base_estimator)

        # 4. filter predictive subsets (ensemble members)
        self.active_subsets_ = [
            stat for stat in subset_stats if stat["score"] >= cutoff
        ]

        # fallback if bootstrap variance excluded even the best model (rare)
        if not self.active_subsets_:
            best_model_stat = max(subset_stats, key=lambda x: x["score"])
            self.active_subsets_ = [best_model_stat]

        # 5. compute weights (uniform averaging)
        n_active = len(self.active_subsets_)
        for stat in self.active_subsets_:
            stat["weight"] = 1.0 / n_active

        self.classes_ = self.active_subsets_[0]["model"].classes_
        return self

    def predict_proba(self, X):
        """Predict class probabilities"""
        check_is_fitted(self)
        X = check_array(X)

        n_samples = X.shape[0]
        sum_proba = np.zeros((n_samples, len(self.classes_)))

        for stat in self.active_subsets_:
            subset = stat["subset"]
            model = stat["model"]
            weight = stat["weight"]

            if len(subset) == 0:
                proba = model.predict_proba(np.zeros((n_samples, 0)))
            else:
                proba = model.predict_proba(X[:, subset])

            proba = self._align_proba(model, proba)
            sum_proba += weight * proba

        return sum_proba

    def predict(self, X):
        """predict class labels with a 0.5 threshold"""
        check_is_fitted(self)
        proba = self.predict_proba(X)

        if len(self.classes_) == 1:
            return np.full(X.shape[0], self.classes_[0])

        # proba has columns aligned with self.classes_ (0, 1) if binary
        # We want the column corresponding to the "positive" class (encoded as 1)
        # Since we validated binary, column 1 is the positive class.
        prob_pos = proba[:, 1]

        predictions_int = (prob_pos > 0.5).astype(int)

        # map back to original labels
        return self.le_.inverse_transform(predictions_int)

    # --- private helpers ---

    def _validate_input(self, X, y, environment):
        """handles DataFrame inputs and sklearn validation"""
        if hasattr(X, "columns") and hasattr(X, "drop"):
            if isinstance(y, str):
                y_col = y
                y = X[y_col].values
                X = X.drop(columns=[y_col])

            if isinstance(environment, str):
                env_col = environment
                environment = X[env_col].values
                X = X.drop(columns=[env_col])

        if y is None:
            # if y wasn't a string inside X, it must be provided
            raise ValueError("Target y must be provided.")

        X, y = check_X_y(X, y)

        if environment is None:
            raise ValueError("Environment labels must be provided.")
        environment = check_array(environment, ensure_2d=False, dtype=None)

        return X, y, environment

    def _find_invariant_subsets(
        self, X, y, environment, n_features, inv_test, base_estimator
    ):
        subset_stats = []
        all_p_values = []

        all_indices = range(n_features)
        feature_subsets = chain.from_iterable(
            combinations(all_indices, r) for r in range(0, n_features + 1)
        )

        for subset in feature_subsets:
            subset = list(subset)
            X_S = X[:, subset] if len(subset) > 0 else np.zeros((X.shape[0], 0))

            p_value = inv_test.test(X_S, y, environment)
            all_p_values.append((subset, p_value))

            if self.verbose:
                if self.verbose > 1:
                    logger.debug(f"Subset {subset}: p-value={p_value:.4f}")

            if p_value >= self.alpha_inv:
                model = self._fit_subset_model(X_S, y, base_estimator)
                score = self._compute_pred_score(model, X_S, y)
                subset_stats.append(
                    {
                        "subset": subset,
                        "p_value": p_value,
                        "score": score,
                        "model": model,
                    }
                )
                if self.verbose:
                    logger.info(
                        f"Subset {subset} is invariant (p={p_value:.4f}). Score={score:.4f}"
                    )

        return subset_stats, all_p_values

    def _fit_subset_model(self, X_S, y, base_estimator):
        if X_S.shape[1] == 0:
            model = _EmptySetClassifier()
        else:
            model = clone(base_estimator)
            if isinstance(model, RandomForestClassifier):
                model.set_params(oob_score=True)
        model.fit(X_S, y)
        return model

    def _apply_fallback(self, X, y, all_p_values, base_estimator):
        best_subset, max_p = max(all_p_values, key=lambda x: x[1])
        X_S = X[:, best_subset] if len(best_subset) > 0 else np.zeros((X.shape[0], 0))
        model = self._fit_subset_model(X_S, y, base_estimator)
        score = self._compute_pred_score(model, X_S, y)
        return [
            {"subset": best_subset, "p_value": max_p, "score": score, "model": model}
        ]

    def _compute_cutoff(self, X, y, subset_stats, base_estimator):
        if not subset_stats:
            return -np.inf

        S_max = max(subset_stats, key=lambda x: x["score"])["subset"]
        bootstrap_scores = []

        n_samples = X.shape[0]
        # bootstrap loop
        for _ in range(self.n_bootstrap):
            # 1. resample
            indices = resample(
                np.arange(n_samples), replace=True, random_state=self.random_state_
            )
            # fix for type checking: explicitly cast to array
            indices = np.asarray(indices)
            oob_indices = np.setdiff1d(np.arange(n_samples), indices)

            if len(oob_indices) == 0:
                continue  # skip if full sample

            X_train, y_train = X[indices], y[indices]
            X_test, y_test = X[oob_indices], y[oob_indices]

            # 2. fit on bootstrap
            X_S_train = (
                X_train[:, S_max] if len(S_max) > 0 else np.zeros((len(indices), 0))
            )
            X_S_test = (
                X_test[:, S_max] if len(S_max) > 0 else np.zeros((len(oob_indices), 0))
            )

            if len(S_max) == 0:
                bs_model = _EmptySetClassifier()
            else:
                bs_model = clone(base_estimator)
                if isinstance(bs_model, RandomForestClassifier):
                    # disable internal OOB since we use explicit hold-out
                    bs_model.set_params(oob_score=False)

            bs_model.fit(X_S_train, y_train)

            # 3. score on hold-out
            score = self._compute_pred_score(bs_model, X_S_test, y_test, use_oob=False)
            bootstrap_scores.append(score)

        if not bootstrap_scores:
            return -np.inf

        return np.quantile(bootstrap_scores, self.alpha_pred)

    def _compute_pred_score(self, model, X, y, use_oob: bool = True):
        """compute negative log loss (higher is better)"""
        proba = self._get_model_proba(model, X, use_oob=use_oob)
        # proba is already aligned to 0,1 classes via _align_proba in _get_model_proba

        if len(self.classes_) == 1:
            # Degenerate case: only one class present
            eps = 1e-15
            col_idx = 0
            y_prob = np.clip(proba[:, col_idx], eps, 1 - eps)
            return -np.mean(np.log(y_prob))

        # y is 0/1 integers here
        return -log_loss(y, proba, labels=[0, 1])

    def _get_model_proba(self, model, X, use_oob: bool) -> np.ndarray:
        if use_oob and hasattr(model, "oob_decision_function_"):
            proba = model.oob_decision_function_
            if np.isnan(proba).any():
                fallback = model.predict_proba(X)
                proba = np.where(np.isnan(proba), fallback, proba)
        else:
            proba = model.predict_proba(X)
        return self._align_proba(model, proba)

    def _align_proba(self, model, proba: np.ndarray) -> np.ndarray:
        """Align a model's probability outputs to self.classes_."""
        # The internal model is trained on integers 0..K-1.
        # Its classes_ are a subset of integers.
        # Each column i in proba corresponds to class model.classes_[i]
        # We need to map this to the column index in 'aligned'.

        aligned = np.zeros((proba.shape[0], len(self.classes_)))
        for i, class_int in enumerate(model.classes_):
            # class_int is 0 or 1.
            # Since we used LabelEncoder, integer 0 is at index 0 of self.classes_
            # and integer 1 is at index 1.
            aligned[:, int(class_int)] = proba[:, i]

        return aligned
