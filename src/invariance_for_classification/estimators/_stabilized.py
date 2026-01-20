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

logger = logging.getLogger(__name__)


def _fit_model_helper(X: np.ndarray, y: np.ndarray, pred_classifier):
    """Helper to fit a model on a subset of features (handles empty sets)."""
    if X.shape[1] == 0:
        model = _EmptySetClassifier()
    else:
        model = clone(pred_classifier)

    model.fit(X, y)
    return model


def _compute_score_helper(
    model, X: np.ndarray, y: np.ndarray, use_oob: bool = True
) -> float:
    """Compute BCE score (negative log loss)."""
    if use_oob and hasattr(model, "oob_decision_function_"):
        y_pred = model.oob_decision_function_
        if np.isnan(y_pred).any():
            fallback = model.predict_proba(X)
            y_pred = np.where(np.isnan(y_pred), fallback, y_pred)
    else:
        y_pred = model.predict_proba(X)

    return float(-log_loss(y, y_pred, labels=[0, 1]))


def _subset_worker(subset, X, y, environment, inv_test, pred_classifier, alpha_inv):
    """Worker function to test a subset for invariance and fit model if successful."""
    subset = list(subset)
    X_S = X[:, subset] if len(subset) > 0 else np.zeros((X.shape[0], 0))

    p_value = inv_test.test(X_S, y, environment)

    stat = None
    if p_value >= alpha_inv:
        # fit model
        model = clone(pred_classifier)
        score = None

        if isinstance(model, RandomForestClassifier):
            model.set_params(oob_score=True)
            model = _fit_model_helper(X_S, y, model)
            score = _compute_score_helper(model, X_S, y, use_oob=True)
        else:
            # use CV for score
            cv_proba = cross_val_predict(
                clone(model), X_S, y, cv=5, method="predict_proba", n_jobs=1
            )
            score = float(-log_loss(y, cv_proba, labels=[0, 1]))

            # fit the final model on all data
            model = _fit_model_helper(X_S, y, model)

        stat = {
            "subset": subset,
            "p_value": p_value,
            "score": score,
            "model": model,
        }

    return subset, p_value, stat


def _bootstrap_worker(seed, X, y, S_max, pred_classifier):
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
    bs_model = clone(pred_classifier)
    if isinstance(bs_model, RandomForestClassifier):
        bs_model.set_params(oob_score=False)

    bs_model = _fit_model_helper(X_S_train, y_train, bs_model)

    # predictiveness score (use_oob=False because we have explicit hold-out set)
    score = _compute_score_helper(bs_model, X_S_test, y_test, use_oob=False)

    return score


class _EmptySetClassifier(BaseEstimator, ClassifierMixin):
    """Internal dummy classifier for the empty feature set (use mean of class 1)."""

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
        return (proba[:, 1] >= 0.5).astype(int)


class StabilizedClassificationClassifier(ClassifierMixin, BaseEstimator):
    """
    Stabilized classification classifier

    Ensemble method that averages predictions from multiple classifiers
    trained on subsets of features that are "invariant" and "predictive"

    Algorithm:
    1. iterate over all feature subsets, testing for invariance using `invariance_test`
    2. for invariant subsets, measure predictive performance (binary cross-entropy)
    3. determine a predictive cutoff using bootstrapping on the single best invariant subset
    4. form an ensemble of all invariant subsets exceeding this predictive cutoff

    Parameters
    ----------
    alpha_inv : float, default=0.05
        significance level for the invariance test. Subsets with p-value >= alpha_inv
        are considered invariant

    alpha_pred : float, default=0.05
        parameter controlling the predictive score cutoff (related to the quantile
        of the bootstrap distribution of the best model's performance)

    pred_classifier_type : str, default="RF"
        Classifier type to use for making predictions.
        "RF" for random forest, "LR" for logistic regression.
        Only used if estimator is None.

    test_classifier_type : str, default="RF"
        Classifier type to use for the invariance test.
        "RF" for random forest, "LR" for logistic regression.
        Only used if `invariance_test` is None.

    invariance_test : str, default="inv_residual"
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
        pred_classifier_type: str = "RF",
        test_classifier_type: str = "RF",
        invariance_test: str = "inv_residual",
        n_bootstrap: int = 100,
        verbose: int = 0,
        random_state: Optional[int] = None,
        n_jobs: int = 10,
    ):
        self.alpha_inv = alpha_inv
        self.alpha_pred = alpha_pred
        self.pred_classifier_type = pred_classifier_type
        self.test_classifier_type = test_classifier_type
        self.invariance_test = invariance_test
        self.n_bootstrap = n_bootstrap
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

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

        # determine pred_classifier and invariance test
        if self.pred_classifier_type == "RF":
            pred_classifier = RandomForestClassifier(
                n_estimators=100,
                oob_score=True,
                random_state=self.random_state,
                n_jobs=1,
            )
        elif self.pred_classifier_type == "LR":
            pred_classifier = LogisticRegression(random_state=self.random_state)
        else:
            raise ValueError(
                f"Unknown pred_classifier_type: {self.pred_classifier_type}"
            )

        if self.invariance_test == "inv_residual":
            from ..invariance_tests import InvariantResidualDistributionTest

            inv_test = InvariantResidualDistributionTest(
                test_classifier_type=self.test_classifier_type
            )
        elif self.invariance_test == "tram_gcm":
            from ..invariance_tests import TramGcmTest

            inv_test = TramGcmTest(test_classifier_type=self.test_classifier_type)
        else:
            raise ValueError(f"Unknown invariance_test: {self.invariance_test}")

        # 1. determine invariant subsets
        subset_stats, all_p_values = self._find_invariant_subsets(
            X, y_encoded, environment, self.n_features_in_, inv_test, pred_classifier
        )

        # fallback if no invariant subsets found
        if not subset_stats:
            if self.verbose:
                logger.warning(
                    "No invariant subsets found. Using subset with max p-value."
                )
            subset_stats = self._apply_no_inv_fallback(
                X, y_encoded, all_p_values, pred_classifier
            )

        # 2. compute predictive cutoff via bootstrap
        cutoff = self._compute_cutoff(X, y_encoded, subset_stats, pred_classifier)

        # 3. filter predictive subsets (ensemble members)
        self.active_subsets_ = [
            stat for stat in subset_stats if stat["score"] >= cutoff
        ]

        # fallback if bootstrap variance excluded best model (rare)
        if not self.active_subsets_:
            best_model_stat = max(subset_stats, key=lambda x: x["score"])
            self.active_subsets_ = [best_model_stat]

        # 4. compute weights (uniform averaging)
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

            proba = model.predict_proba(X_tilde)
            sum_proba += weight * proba

        return sum_proba

    def predict(self, X: np.ndarray, threshold: float = 0.5):
        """predict class labels with a threshold (default: 0.5)"""
        check_is_fitted(self)
        X = self._validate_X(X)
        proba = self.predict_proba(X)

        prob_pos = proba[:, 1]

        predictions_int = (prob_pos > threshold).astype(int)

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

        if len(np.unique(environment)) < 2:
            raise ValueError(
                f"Validation Error: Environment variable must contain at least 2 unique values. Found {len(np.unique(environment))} ({np.unique(environment)})."
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
        self, X, y, environment, n_features, inv_test, pred_classifier
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
                    pred_classifier,
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

    def _apply_no_inv_fallback(self, X, y, all_p_values, pred_classifier):
        """Fallback to the subset with the highest p-value in case no invariant subsets found."""
        best_subset, max_p = max(all_p_values, key=lambda x: x[1])
        X_S = X[:, best_subset] if len(best_subset) > 0 else np.zeros((X.shape[0], 0))

        model = clone(pred_classifier)
        score = None
        if isinstance(model, RandomForestClassifier):
            model.set_params(oob_score=True)
            model = _fit_model_helper(X_S, y, model)
            score = _compute_score_helper(model, X_S, y, use_oob=True)
        else:
            cv_proba = cross_val_predict(
                clone(model), X_S, y, cv=5, method="predict_proba", n_jobs=1
            )
            score = float(-log_loss(y, cv_proba, labels=[0, 1]))
            model = _fit_model_helper(X_S, y, model)

        return [
            {
                "subset": best_subset,
                "p_value": max_p,
                "score": score,
                "model": model,
            }
        ]

    def _compute_cutoff(self, X, y, subset_stats, pred_classifier):
        """Compute the predictive cutoff score via bootstrapping the best subset."""
        if not subset_stats:
            return -np.inf

        S_max = max(subset_stats, key=lambda x: x["score"])["subset"]

        # generate seeds for each bootstrap iteration
        seeds = self.random_state_.randint(
            0, np.iinfo(np.int32).max, size=self.n_bootstrap
        )

        bootstrap_scores = Parallel(n_jobs=self.n_jobs)(
            delayed(_bootstrap_worker)(seed, X, y, S_max, pred_classifier)
            for seed in seeds
        )

        # filter None values (if oob_indices was empty)
        bootstrap_scores = [s for s in bootstrap_scores if s is not None]

        if not bootstrap_scores:
            return -np.inf

        return np.quantile(bootstrap_scores, self.alpha_pred)
