import logging
from itertools import chain, combinations
from typing import Any, Optional, Union, cast

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


def _aggregate_score(y, y_pred, environment, pred_scoring):
    """Compute negative log-loss, either pooled or worst-case across environments."""
    if pred_scoring == "pooled":
        return float(-log_loss(y, y_pred, labels=[0, 1]))
    elif pred_scoring == "worst_case":
        env_scores = []
        for e in np.unique(environment):
            mask = environment == e
            if len(np.unique(y[mask])) < 2:
                continue
            env_scores.append(float(-log_loss(y[mask], y_pred[mask], labels=[0, 1])))
        return min(env_scores) if env_scores else -np.inf
    else:
        raise ValueError(f"Unknown pred_scoring: {pred_scoring}")


def _fit_and_score(X_S, y, environment, pred_classifier, pred_scoring="pooled"):
    """Fit a model on all data and compute its out-of-sample predictiveness score.

    Returns (model, score) where model is fitted on ALL data.
    """
    # Handle empty feature set (0 columns) – use majority-class dummy
    if X_S.shape[1] == 0:
        model = _EmptySetClassifier()
        model.fit(X_S, y)
        y_pred = model.predict_proba(X_S)
        score = _aggregate_score(y, y_pred, environment, pred_scoring)
        return model, score

    is_rf = isinstance(pred_classifier, RandomForestClassifier)

    # get out-of-sample predictions and fitted model
    if is_rf:
        model = clone(pred_classifier)
        model.set_params(oob_score=True)
        model = _fit_model_helper(X_S, y, model)
        y_pred = model.oob_decision_function_
        if np.isnan(y_pred).any():
            fallback = model.predict_proba(X_S)
            y_pred = np.where(np.isnan(y_pred), fallback, y_pred)
    else:
        y_pred = cross_val_predict(
            clone(pred_classifier),
            X_S,
            y,
            cv=5,
            method="predict_proba",
            n_jobs=1,
        )
        model = _fit_model_helper(X_S, y, pred_classifier)

    score = _aggregate_score(y, y_pred, environment, pred_scoring)
    return model, score


# ---------------------------------------------------------------------------
# Worker functions for parallel execution
# ---------------------------------------------------------------------------


def _subset_worker(
    subset, X, y, environment, inv_test, pred_classifiers, alpha_inv, pred_scoring
):
    """Compute p-value and, if invariant, fit all prediction classifiers.

    Parameters
    ----------
    pred_classifiers : dict[str, estimator]
        Mapping of classifier name (e.g. "RF", "LR") to sklearn estimator.

    Returns
    -------
    dict with keys: subset, p_value, and for each invariant classifier
    ``{name}_model`` and ``{name}_score``.
    """
    subset = list(subset)
    X_S = X[:, subset] if len(subset) > 0 else np.zeros((X.shape[0], 0))
    p_value = inv_test.test(X_S, y, environment)
    result: dict[str, Any] = {"subset": subset, "p_value": p_value}

    if p_value >= alpha_inv:
        for name, clf in pred_classifiers.items():
            model, score = _fit_and_score(X_S, y, environment, clf, pred_scoring)
            result[f"{name}_model"] = model
            result[f"{name}_score"] = score

    return result


def _loeo_subset_worker(
    subset,
    X,
    y,
    environment,
    loeo_ranking_fn,
    loeo_classifier_type,
    loeo_random_state,
    pred_classifiers,
    pred_scoring,
):
    """Compute LOEO score and fit all prediction classifiers.

    Models are always fitted (the invariance cutoff is not known yet).

    Returns
    -------
    dict with keys: subset, inv_score, and ``{name}_model`` / ``{name}_score``
    for each classifier in *pred_classifiers*.
    """
    subset = list(subset)
    X_S = X[:, subset] if len(subset) > 0 else np.zeros((X.shape[0], 0))
    scores = loeo_ranking_fn(
        Y=y,
        E=environment,
        X_S=X_S,
        classifier_type=loeo_classifier_type,
        random_state=loeo_random_state,
    )
    result: dict[str, Any] = {"subset": subset, "inv_score": scores["mean"]}

    for name, clf in pred_classifiers.items():
        model, score = _fit_and_score(X_S, y, environment, clf, pred_scoring)
        result[f"{name}_model"] = model
        result[f"{name}_score"] = score

    return result


def _bootstrap_worker(
    seed, X, y, S_max, pred_classifier, pred_scoring="pooled", environment=None
):
    """Worker function to compute a single bootstrap score."""
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]
    indices = np.asarray(resample(np.arange(n_samples), replace=True, random_state=rng))
    oob_indices = np.setdiff1d(np.arange(n_samples), indices)

    if len(oob_indices) == 0:
        return None

    X_S_train = X[indices][:, S_max] if len(S_max) > 0 else np.zeros((len(indices), 0))
    X_S_test = (
        X[oob_indices][:, S_max] if len(S_max) > 0 else np.zeros((len(oob_indices), 0))
    )

    bs_model = clone(pred_classifier)
    if isinstance(bs_model, RandomForestClassifier):
        bs_model.set_params(oob_score=False)
    bs_model = _fit_model_helper(X_S_train, y[indices], bs_model)

    y_pred = bs_model.predict_proba(X_S_test)
    env_oob = environment[oob_indices] if environment is not None else None
    return _aggregate_score(y[oob_indices], y_pred, env_oob, pred_scoring)


def _loeo_regret_inv_bootstrap_worker(
    seed,
    X,
    y,
    environment,
    S_best,
    loeo_ranking_fn,
    loeo_classifier_type,
    loeo_random_state,
):
    """Worker to bootstrap a single LOEO ranking score for invariance cutoff."""
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]
    indices = np.asarray(resample(np.arange(n_samples), replace=True, random_state=rng))

    X_boot = X[indices]
    y_boot = y[indices]
    env_boot = environment[indices]

    # need at least 2 environments for meaningful ranking
    if len(np.unique(env_boot)) < 2:
        return None

    X_S_boot = X_boot[:, S_best] if len(S_best) > 0 else np.zeros((len(indices), 0))

    scores = loeo_ranking_fn(
        Y=y_boot,
        E=env_boot,
        X_S=X_S_boot,
        classifier_type=loeo_classifier_type,
        random_state=loeo_random_state,
    )

    return scores["mean"]


# ---------------------------------------------------------------------------
# Helper classifiers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


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
        For statistical invariance tests: significance level. Subsets with
        p-value >= alpha_inv are considered invariant.
        For LOEO-regret ranking: quantile of the bootstrap distribution of the best
        subset's invariance score used as the cutoff. Subsets with invariance
        score >= this cutoff are kept.

    alpha_pred : float, default=0.05
        parameter controlling the predictive score cutoff (related to the quantile
        of the bootstrap distribution of the best model's performance)

    pred_classifier_type : str or list[str], default="RF"
        Classifier type(s) to use for making predictions.
        "RF" for random forest, "LR" for logistic regression.
        When a list is given (e.g. ``["RF", "LR"]``), invariance scores are
        computed once and prediction models are fitted for each type
        separately. Use the ``pred_classifier_type`` argument in
        ``predict`` / ``predict_proba`` to select which classifier's
        predictions to use.

    test_classifier_type : str, default="RF"
        Classifier type to use for the invariance test.
        "RF" for random forest, "LR" for logistic regression.
        Passed to the invariance test's ``test_classifier_type`` parameter.

    invariance_test : str, default="inv_residual"
        The invariance test or ranking method to use. Options:
        - "inv_residual": InvariantResidualDistributionTest
        - "tram_gcm": TramGcmTest
        - "wgcm": WGCMTest
        - "delong": DeLongTest
        - "inv_env_pred": InvariantEnvironmentPredictionTest
        - "crt": ConditionalRandomizationTest
        - "loeo_regret": Uses LOEO regret ranking (not a statistical test). Subsets
          are ranked by their LOEO regret score and filtered using a
          bootstrap-based cutoff instead of a p-value threshold.

    pred_scoring : str, default="pooled"
        Strategy for computing the predictiveness score of invariant subsets.
        - "pooled": Train and evaluate on pooled data across all training
          environments (standard ERM). This is the default.
        - "worst_case": Train on all environments pooled, but evaluate
          per-environment and take the worst (maximum) risk across
          environments.

    n_bootstrap : int, default=250
        number of bootstrap samples used to determine the predictive cutoff

    verbose : int, default=0
        verbosity level

    random_state : int, RandomState instance or None, default=None
        random state of the estimator

    Attributes (after fitting)
    --------------------------
    n_subsets_total_ : int
        Total number of feature subsets considered (2^p).
    n_invariant_subsets_ : int
        Number of subsets that passed the invariance filter.
    n_predictive_subsets_ : int or dict[str, int]
        Number of subsets in the final predictive ensemble.
        ``int`` when ``pred_classifier_type`` is a string,
        ``dict`` keyed by classifier type when it is a list.
    active_subsets_ : list[dict] or dict[str, list[dict]]
        The ensemble members.  ``list`` when ``pred_classifier_type`` is a
        string, ``dict`` keyed by classifier type when it is a list.
    """

    def __init__(
        self,
        alpha_inv: float = 0.05,
        alpha_pred: float = 0.05,
        pred_classifier_type: Union[str, list[str]] = "RF",
        test_classifier_type: str = "RF",
        invariance_test: str = "inv_residual",
        pred_scoring: str = "pooled",
        n_bootstrap: int = 250,
        verbose: int = 0,
        random_state: Optional[int] = None,
        n_jobs: int = 10,
    ):
        self.alpha_inv = alpha_inv
        self.alpha_pred = alpha_pred
        self.pred_classifier_type = pred_classifier_type
        self.test_classifier_type = test_classifier_type
        self.invariance_test = invariance_test
        self.pred_scoring = pred_scoring
        self.n_bootstrap = n_bootstrap
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        n_features = self.n_features_in_

        # Normalize pred_classifier_type to a list for unified processing
        if isinstance(self.pred_classifier_type, list):
            self._multi_classifier = True
            clf_types: list[str] = self.pred_classifier_type
        else:
            self._multi_classifier = False
            clf_types: list[str] = [self.pred_classifier_type]
        pred_classifiers = {ct: self._make_pred_classifier(ct) for ct in clf_types}

        if self.pred_scoring not in ("pooled", "worst_case"):
            raise ValueError(
                f"Unknown pred_scoring: {self.pred_scoring}. "
                "Choose from 'pooled', 'worst_case'."
            )

        # Generate all 2^p feature subsets
        all_indices = range(n_features)
        feature_subsets = list(
            chain.from_iterable(
                combinations(all_indices, r) for r in range(0, n_features + 1)
            )
        )

        # --- Step 1: Evaluate all subsets (invariance + fit classifiers) ---
        if self.invariance_test == "loeo_regret":
            from ..rankings import loeo_regret

            all_results = cast(
                list[dict[str, Any]],
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_loeo_subset_worker)(
                        subset,
                        X,
                        y_encoded,
                        environment,
                        loeo_regret,
                        self.test_classifier_type,
                        self.random_state,
                        pred_classifiers,
                        self.pred_scoring,
                    )
                    for subset in feature_subsets
                ),
            )

            if self.verbose:
                for r in all_results:
                    if self.verbose > 1:
                        logger.debug(
                            f"Subset {r['subset']}: loeo_score={r['inv_score']:.6f}"
                        )

            # Compute invariance cutoff via bootstrap, then filter
            inv_cutoff = self._compute_invariance_cutoff(
                X, y_encoded, environment, all_results
            )
            invariant_results = [r for r in all_results if r["inv_score"] >= inv_cutoff]
            if not invariant_results:
                if self.verbose:
                    logger.warning(
                        "No subsets above invariance cutoff. "
                        "Using subset with highest LOEO regret score."
                    )
                invariant_results = [max(all_results, key=lambda x: x["inv_score"])]
        else:
            inv_test = self._setup_inv_test()

            all_results = cast(
                list[dict[str, Any]],
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_subset_worker)(
                        subset,
                        X,
                        y_encoded,
                        environment,
                        inv_test,
                        pred_classifiers,
                        self.alpha_inv,
                        self.pred_scoring,
                    )
                    for subset in feature_subsets
                ),
            )

            if self.verbose:
                for r in all_results:
                    if self.verbose > 1:
                        logger.debug(
                            f"Subset {r['subset']}: p-value={r['p_value']:.4f}"
                        )
                    if r["p_value"] >= self.alpha_inv:
                        logger.info(
                            f"Subset {r['subset']} is invariant (p={r['p_value']:.4f})"
                        )

            invariant_results = [
                r for r in all_results if r["p_value"] >= self.alpha_inv
            ]
            if not invariant_results:
                if self.verbose:
                    logger.warning(
                        "No invariant subsets found. Using subset with max p-value."
                    )
                # Fallback: pick best p-value subset and fit classifiers for it
                best = dict(max(all_results, key=lambda x: x["p_value"]))
                subset = best["subset"]
                X_S = X[:, subset] if len(subset) > 0 else np.zeros((X.shape[0], 0))
                for name, clf in pred_classifiers.items():
                    model, score = _fit_and_score(
                        X_S, y_encoded, environment, clf, self.pred_scoring
                    )
                    best[f"{name}_model"] = model
                    best[f"{name}_score"] = score
                invariant_results = [best]

        self.n_subsets_total_ = 2**n_features
        self.n_invariant_subsets_ = len(invariant_results)

        # --- Step 2: Per classifier – predictive cutoff and ensemble filtering ---
        all_fitted_by_clf: dict[str, list[dict[str, Any]]] = {}
        active_by_clf: dict[str, list[dict[str, Any]]] = {}
        n_pred_by_clf: dict[str, int] = {}

        for ct, clf in pred_classifiers.items():
            # Build per-classifier list from the shared invariant results
            subset_stats: list[dict[str, Any]] = [
                {
                    "subset": r["subset"],
                    "score": r[f"{ct}_score"],
                    "model": r[f"{ct}_model"],
                }
                for r in invariant_results
            ]

            cutoff = self._compute_cutoff(X, y_encoded, environment, subset_stats, clf)
            active = [s for s in subset_stats if s["score"] >= cutoff]
            if not active:
                active = [max(subset_stats, key=lambda x: x["score"])]
            n_active = len(active)
            for s in active:
                s["weight"] = 1.0 / n_active

            all_fitted_by_clf[ct] = subset_stats
            active_by_clf[ct] = active
            n_pred_by_clf[ct] = n_active

        # Store results — keep backward-compatible types for single classifier
        if self._multi_classifier:
            self.active_subsets_ = active_by_clf
            self.n_predictive_subsets_ = n_pred_by_clf
            self._all_invariant_fitted_ = all_fitted_by_clf
        else:
            ct0 = clf_types[0]
            self.active_subsets_ = active_by_clf[ct0]
            self.n_predictive_subsets_ = n_pred_by_clf[ct0]
            self._all_invariant_fitted_ = all_fitted_by_clf[ct0]

        return self

    def predict_proba(self, X, pred_classifier_type=None, method="ensemble"):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        pred_classifier_type : str or None, default=None
            Which classifier's predictions to use.  Required when
            ``pred_classifier_type`` was set to a list during ``__init__``.
            Ignored (uses the single fitted type) when a string was given.
        method : {"ensemble", "best"}, default="ensemble"
            - "ensemble": average predictions from all active subsets (default).
            - "best": use only the single most predictive invariant subset.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
        """
        check_is_fitted(self)
        X = self._validate_X(X)

        active, all_fitted = self._get_active_subsets(pred_classifier_type)

        if method == "best":
            best = max(all_fitted, key=lambda x: x["score"])
            active = [{**best, "weight": 1.0}]
        elif method != "ensemble":
            raise ValueError(f"Unknown method: {method}. Choose 'ensemble' or 'best'.")

        n_samples = X.shape[0]
        sum_proba = np.zeros((n_samples, 2), dtype=float)

        for stat in active:
            subset = stat["subset"]
            model = stat["model"]
            weight = stat["weight"]

            X_tilde = X[:, subset] if len(subset) > 0 else np.zeros((n_samples, 0))

            proba = model.predict_proba(X_tilde)
            sum_proba += weight * proba

        return sum_proba

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
        pred_classifier_type=None,
        method="ensemble",
    ):
        """Predict class labels.

        Parameters
        ----------
        X : array-like
            Input data.
        threshold : float, default=0.5
            Decision threshold on P(Y=1).
        pred_classifier_type : str or None
            See ``predict_proba``.
        method : {"ensemble", "best"}
            See ``predict_proba``.
        """
        check_is_fitted(self)
        X = self._validate_X(X)
        proba = self.predict_proba(
            X, pred_classifier_type=pred_classifier_type, method=method
        )

        prob_pos = proba[:, 1]

        predictions_int = (prob_pos > threshold).astype(int)

        # map back to original labels
        return self.le_.inverse_transform(predictions_int)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

    def _get_active_subsets(
        self, pred_classifier_type: Optional[str] = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (active_subsets, all_invariant_fitted) for a classifier type.

        For a single-classifier fit, returns the stored lists directly.
        For a multi-classifier fit, indexes into the per-classifier dicts.
        """
        if self._multi_classifier:
            assert isinstance(self.active_subsets_, dict)
            assert isinstance(self._all_invariant_fitted_, dict)
            if pred_classifier_type is None:
                raise ValueError(
                    "pred_classifier_type must be specified when multiple "
                    "classifier types were fitted. "
                    f"Available: {list(self.active_subsets_.keys())}"
                )
            if pred_classifier_type not in self.active_subsets_:
                raise ValueError(
                    f"Unknown pred_classifier_type: {pred_classifier_type}. "
                    f"Available: {list(self.active_subsets_.keys())}"
                )
            return (
                self.active_subsets_[pred_classifier_type],
                self._all_invariant_fitted_[pred_classifier_type],
            )
        else:
            assert isinstance(self.active_subsets_, list)
            assert isinstance(self._all_invariant_fitted_, list)
            return self.active_subsets_, self._all_invariant_fitted_

    def _make_pred_classifier(self, clf_type: str):
        """Create a prediction classifier instance by type string."""
        if clf_type == "RF":
            return RandomForestClassifier(
                n_estimators=100,
                oob_score=True,
                random_state=self.random_state,
                n_jobs=1,
            )
        elif clf_type == "LR":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            random_state=self.random_state, max_iter=1000
                        ),
                    ),
                ]
            )
        else:
            raise ValueError(
                f"Unknown pred_classifier_type: {clf_type}. Choose from 'RF', 'LR'."
            )

    def _setup_inv_test(self):
        """Create the invariance test object based on self.invariance_test."""
        if self.invariance_test == "inv_residual":
            from ..invariance_tests import InvariantResidualDistributionTest

            return InvariantResidualDistributionTest(
                test_classifier_type=self.test_classifier_type
            )
        elif self.invariance_test == "tram_gcm":
            from ..invariance_tests import TramGcmTest

            return TramGcmTest(test_classifier_type=self.test_classifier_type)
        elif self.invariance_test == "wgcm":
            from ..invariance_tests import WGCMTest

            return WGCMTest(use_categorical_loss=True)
        elif self.invariance_test == "delong":
            from ..invariance_tests import DeLongTest

            return DeLongTest(test_classifier_type=self.test_classifier_type)
        elif self.invariance_test == "inv_env_pred":
            from ..invariance_tests import InvariantEnvironmentPredictionTest

            return InvariantEnvironmentPredictionTest(
                test_classifier_type=self.test_classifier_type
            )
        elif self.invariance_test == "crt":
            from ..invariance_tests import ConditionalRandomizationTest

            return ConditionalRandomizationTest(
                test_classifier_type=self.test_classifier_type
            )
        else:
            raise ValueError(f"Unknown invariance_test: {self.invariance_test}")

    def _compute_cutoff(self, X, y, environment, subset_stats, pred_classifier):
        """Compute the predictive cutoff score via bootstrapping the best subset."""
        if not subset_stats:
            return -np.inf

        S_max = max(subset_stats, key=lambda x: x["score"])["subset"]

        # generate seeds for each bootstrap iteration
        seeds = self.random_state_.randint(
            0, np.iinfo(np.int32).max, size=self.n_bootstrap
        )

        bootstrap_scores = Parallel(n_jobs=self.n_jobs)(
            delayed(_bootstrap_worker)(
                seed, X, y, S_max, pred_classifier, self.pred_scoring, environment
            )
            for seed in seeds
        )

        # filter None values (if oob_indices was empty)
        bootstrap_scores = [s for s in bootstrap_scores if s is not None]

        if not bootstrap_scores:
            return -np.inf

        return np.quantile(bootstrap_scores, self.alpha_pred)

    def _compute_invariance_cutoff(self, X, y, environment, all_stats):
        """Compute invariance cutoff via bootstrapping the best LOEO-ranked subset."""
        from ..rankings import loeo_regret

        if not all_stats:
            return -np.inf

        S_best = max(all_stats, key=lambda x: x["inv_score"])["subset"]

        seeds = self.random_state_.randint(
            0, np.iinfo(np.int32).max, size=self.n_bootstrap
        )

        bootstrap_scores = Parallel(n_jobs=self.n_jobs)(
            delayed(_loeo_regret_inv_bootstrap_worker)(
                seed,
                X,
                y,
                environment,
                S_best,
                loeo_regret,
                self.test_classifier_type,
                self.random_state,
            )
            for seed in seeds
        )

        # filter None values (if bootstrap sample had < 2 environments)
        bootstrap_scores = [s for s in bootstrap_scores if s is not None]

        if not bootstrap_scores:
            return -np.inf

        return np.quantile(bootstrap_scores, self.alpha_inv)
