import logging
from itertools import chain, combinations
from typing import Optional, cast

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state, resample
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..invariance_tests import InvarianceTest, InvariantResidualDistributionTest

logger = logging.getLogger(__name__)


def _fit_model_helper(
    X: np.ndarray, y: np.ndarray, base_estimator, force_n_jobs_1: bool = False
):
    """
    Helper to fit a model on a subset of features.
    Handles empty sets and n_jobs configuration.
    """
    if X.shape[1] == 0:
        model = _EmptySetClassifier()
    else:
        model = clone(base_estimator)
        if force_n_jobs_1 and hasattr(model, "n_jobs"):
            model.set_params(n_jobs=1)

    model.fit(X, y)
    return model


def _get_aligned_proba(model, X: np.ndarray, use_oob: bool = True) -> np.ndarray:
    """
    Get probability estimates aligned to encoded labels [0, 1].

    Returns an array of shape (n_samples, 2) where column 0 corresponds to the
    encoded class 0 and column 1 corresponds to encoded class 1.
    """
    if use_oob and hasattr(model, "oob_decision_function_"):
        proba = model.oob_decision_function_
        if np.isnan(proba).any():
            fallback = model.predict_proba(X)
            proba = np.where(np.isnan(proba), fallback, proba)
    else:
        proba = model.predict_proba(X)

    proba = np.asarray(proba)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    aligned = np.zeros((proba.shape[0], 2), dtype=float)
    classes = getattr(model, "classes_", np.array([0, 1]))
    for i, class_label in enumerate(classes):
        if class_label in (0, 1):
            aligned[:, int(class_label)] = proba[:, i]

    row_sums = aligned.sum(axis=1)
    missing = row_sums == 0
    if np.any(missing):
        aligned[missing] = 0.5
        row_sums[missing] = 1.0
    aligned = aligned / row_sums[:, None]
    return aligned


def _compute_score_helper(
    model, X: np.ndarray, y: np.ndarray, use_oob: bool = True
) -> float:
    """Compute score (negative log loss) for binary classification."""
    # y must be 0/1 integers
    proba = _get_aligned_proba(model, X, use_oob=use_oob)
    return float(-log_loss(y, proba, labels=[0, 1]))


def _subset_worker(subset, X, y, environment, inv_test, base_estimator, alpha_inv):
    """Worker function to test a subset for invariance and fit model if successful."""
    subset = list(subset)
    X_S = X[:, subset] if len(subset) > 0 else np.zeros((X.shape[0], 0))

    p_value = inv_test.test(X_S, y, environment)

    stat = None
    if p_value >= alpha_inv:
        # fit model
        model = clone(base_estimator)
        score = None

        if isinstance(model, RandomForestClassifier):
            model.set_params(oob_score=True)
            model = _fit_model_helper(X_S, y, model, force_n_jobs_1=True)
            # compute predictiveness score using OOB
            score = _compute_score_helper(model, X_S, y, use_oob=True)
        else:
            # use CV for score
            try:
                # y is encoded 0/1, so we expect two classes
                cv_proba = cross_val_predict(
                    clone(model), X_S, y, cv=5, method="predict_proba", n_jobs=1
                )
                # cv_proba is (n_samples, 2). Align with y (0/1)
                # score is negative log loss
                score = -log_loss(y, cv_proba, labels=[0, 1])
            except Exception:
                # Fallback to in-sample if CV fails
                pass

            # fit the final model on all data
            model = _fit_model_helper(X_S, y, model, force_n_jobs_1=True)
            if score is None:
                score = _compute_score_helper(model, X_S, y, use_oob=False)

        stat = {
            "subset": subset,
            "p_value": p_value,
            "score": float(score),
            "model": model,
        }

    return subset, p_value, stat


def _bootstrap_worker(seed, X, y, S_max, base_estimator):
    """Worker function to compute a single bootstrap score."""
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]
    indices = np.asarray(resample(np.arange(n_samples), replace=True, random_state=rng))
    oob_indices = np.setdiff1d(np.arange(n_samples), indices)

    if len(oob_indices) == 0:
        return None

    X_train, y_train = X[indices], y[indices]
    X_test, y_test = X[oob_indices], y[oob_indices]

    X_S_train = X_train[:, S_max] if len(S_max) > 0 else np.zeros((len(indices), 0))
    X_S_test = X_test[:, S_max] if len(S_max) > 0 else np.zeros((len(oob_indices), 0))

    # prepare model
    bs_model = clone(base_estimator)
    if isinstance(bs_model, RandomForestClassifier):
        bs_model.set_params(oob_score=False)

    bs_model = _fit_model_helper(X_S_train, y_train, bs_model, force_n_jobs_1=True)

    # predictiveness score (use_oob=False because we have explicit hold-out set)
    score = _compute_score_helper(bs_model, X_S_test, y_test, use_oob=False)

    return score


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


class StabilizedClassificationClassifier(ClassifierMixin, BaseEstimator):
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

    classifier_type : str, default="RF"
        Classifier type to use for the main estimator.
        "RF" for random forest, "LR" for logistic regression.
        Only used if estimator is None.

    test_classifier_type : str, default="LR"
        Classifier type to use for the invariance test.
        "RF" for random forest, "LR" for logistic regression.
        Only used if `invariance_test` is None.

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
        classifier_type: str = "RF",
        test_classifier_type: str = "LR",
        invariance_test: Optional[InvarianceTest] = None,
        n_bootstrap: int = 100,
        verbose: int = 0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.alpha_inv = alpha_inv
        self.alpha_pred = alpha_pred
        self.classifier_type = classifier_type
        self.test_classifier_type = test_classifier_type
        self.invariance_test = invariance_test
        self.n_bootstrap = n_bootstrap
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

    # sklearn model interface
    def _more_tags(self):
        return {"binary_only": True}

    def fit(self, X, y=None, environment=None):
        """
        Fit the model.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data. If a DataFrame is provided, you may pass column names
            for `y` and `environment`; those columns are removed from `X` and the
            remaining columns are used as features.

            Example
            -------
            If ``df`` has columns ``X1, X2, E, Y`` and you do
            ``fit(df, y="Y", environment="E")``, then ``X1`` and ``X2``
            are used as predictors. Columns ``Y`` and ``E`` are extracted
            as target and environment and are not used as predictors.

            Make sure to drop any extra columns that are neither predictors,
            the label, nor the environment.

        y : array-like or str, optional
            Target values (binary) or column name in `X`.

        environment : array-like or str
            Environment labels or column name in `X`. Must be provided.

        Returns
        -------
        self : object
        """
        X, y, environment = self._validate_input(X, y, environment)

        self.random_state_ = check_random_state(self.random_state)

        # encode target y to 0/1 integers
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        if len(self.classes_) != 2:
            raise ValueError(
                "Only binary classification with exactly 2 classes is supported. "
                f"Found classes: {self.classes_}"
            )

        self.n_features_in_ = X.shape[1]

        # determine base estimator and invariance test
        if self.classifier_type == "RF":
            base_estimator = RandomForestClassifier(
                n_estimators=100, oob_score=True, random_state=self.random_state
            )
        elif self.classifier_type == "LR":
            base_estimator = LogisticRegression(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")

        if self.invariance_test is not None:
            inv_test = self.invariance_test
        else:
            # check if user specified a different classifier for the test
            start_test_classifier_type = (
                self.test_classifier_type or self.classifier_type
            )
            inv_test = InvariantResidualDistributionTest(
                classifier_type=start_test_classifier_type
            )

        # keep invariance-test estimator single-threaded to avoid nested parallelism
        inv_estimator = getattr(inv_test, "estimator", None)
        if inv_estimator is not None and hasattr(inv_estimator, "n_jobs"):
            inv_estimator.set_params(n_jobs=1)

        # 1. & 2.: determine invariant subsets
        subset_stats, all_p_values = self._find_invariant_subsets(
            X, y_encoded, environment, self.n_features_in_, inv_test, base_estimator
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

        # fallback if bootstrap variance excluded best model (rare)
        if not self.active_subsets_:
            best_model_stat = max(subset_stats, key=lambda x: x["score"])
            self.active_subsets_ = [best_model_stat]

        # 5. compute weights (uniform averaging)
        n_active = len(self.active_subsets_)
        for stat in self.active_subsets_:
            stat["weight"] = 1.0 / n_active

        return self

    def predict_proba(self, X):
        """Predict class probabilities"""
        check_is_fitted(self)
        X = self._validate_X(X)

        n_samples = X.shape[0]
        sum_proba = np.zeros((n_samples, 2), dtype=float)

        for stat in self.active_subsets_:
            subset = stat["subset"]
            model = stat["model"]
            weight = stat["weight"]

            X_tilde = X[:, subset] if len(subset) > 0 else np.zeros((n_samples, 0))

            proba = _get_aligned_proba(model, X_tilde, use_oob=False)
            sum_proba += weight * proba

        return sum_proba

    def predict(self, X):
        """predict class labels with a 0.5 threshold"""
        check_is_fitted(self)
        X = self._validate_X(X)
        proba = self.predict_proba(X)

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
                y = X[y_col].to_numpy()
                X = X.drop(columns=[y_col])

            if isinstance(environment, str):
                env_col = environment
                environment = X[env_col].to_numpy()
                X = X.drop(columns=[env_col])

        if y is None:
            # if y wasn't a string inside X, it must be provided
            raise ValueError("Target y must be provided.")

        X, y = check_X_y(X, y)

        if environment is None:
            raise ValueError(
                "Environment labels must be provided for stabilized classification."
            )

        environment = check_array(environment, ensure_2d=False, dtype=None)

        unique_envs = np.unique(environment)
        if len(unique_envs) < 2:
            raise ValueError(
                f"Validation Error: Environment variable must contain at least 2 unique values. Found {len(unique_envs)} ({unique_envs})."
            )

        return X, y, environment

    def _validate_X(self, X):
        """Validate input features X against the training feature count."""
        X = check_array(X)
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )
        return X

    def _find_invariant_subsets(
        self, X, y, environment, n_features, inv_test, base_estimator
    ):
        """Iterate over all feature subsets to identify invariant ones."""
        subset_stats = []
        all_p_values = []

        all_indices = range(n_features)
        feature_subsets = chain.from_iterable(
            combinations(all_indices, r) for r in range(0, n_features + 1)
        )

        results = cast(
            list[tuple[list[int], float, Optional[dict]]],
            Parallel(n_jobs=self.n_jobs)(
                delayed(_subset_worker)(
                    subset,
                    X,
                    y,
                    environment,
                    inv_test,
                    base_estimator,
                    self.alpha_inv,
                )
                for subset in feature_subsets
            ),
        )

        for subset, p_value, stat in results:
            all_p_values.append((subset, p_value))

            if self.verbose:
                if self.verbose > 1:
                    logger.debug(f"Subset {subset}: p-value={p_value:.4f}")

            if stat is not None:
                subset_stats.append(stat)
                if self.verbose:
                    logger.info(
                        f"Subset {subset} is invariant (p={p_value:.4f}). Score={stat['score']:.4f}"
                    )

        return subset_stats, all_p_values

    def _apply_fallback(self, X, y, all_p_values, base_estimator):
        """Fallback to the subset with the highest p-value in case no invariant subsets found."""
        best_subset, max_p = max(all_p_values, key=lambda x: x[1])
        X_S = X[:, best_subset] if len(best_subset) > 0 else np.zeros((X.shape[0], 0))

        model = clone(base_estimator)
        score = None
        if isinstance(model, RandomForestClassifier):
            model.set_params(oob_score=True)
            model = _fit_model_helper(X_S, y, model)
            score = _compute_score_helper(model, X_S, y, use_oob=True)
        else:
            try:
                cv_proba = cross_val_predict(
                    clone(model), X_S, y, cv=5, method="predict_proba"
                )
                score = -log_loss(y, cv_proba, labels=[0, 1])
            except Exception:
                pass
            model = _fit_model_helper(X_S, y, model)
            if score is None:
                score = _compute_score_helper(model, X_S, y, use_oob=False)

        return [
            {
                "subset": best_subset,
                "p_value": max_p,
                "score": float(score),
                "model": model,
            }
        ]

    def _compute_cutoff(self, X, y, subset_stats, base_estimator):
        """Compute the predictive cutoff score via bootstrapping the best subset."""
        if not subset_stats:
            return -np.inf

        S_max = max(subset_stats, key=lambda x: x["score"])["subset"]

        # generate seeds for each bootstrap iteration
        seeds = self.random_state_.randint(
            0, np.iinfo(np.int32).max, size=self.n_bootstrap
        )

        bootstrap_scores = Parallel(n_jobs=self.n_jobs)(
            delayed(_bootstrap_worker)(seed, X, y, S_max, base_estimator)
            for seed in seeds
        )

        # filter None values (if oob_indices was empty)
        bootstrap_scores = [s for s in bootstrap_scores if s is not None]

        if not bootstrap_scores:
            return -np.inf

        return np.quantile(bootstrap_scores, self.alpha_pred)
