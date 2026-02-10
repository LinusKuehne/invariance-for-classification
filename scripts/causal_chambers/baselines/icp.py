"""
Invariant Causal Prediction (ICP) baseline using the R package tramicp.

Two variants:
- ICPglmClassifier: uses glmICP (logistic regression-based)
- ICPrfClassifier: uses rangerICP (random forest-based, nonparametric)

Workflow:
1. Run ICP to find the invariant set (intersection of all non-rejected sets)
2. Fit a Random Forest on only those predictors
3. If the invariant set is empty, predict the class proportions
"""

from __future__ import annotations

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import conversion, default_converter, numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

# create a converter that combines default + numpy + pandas
_converter = default_converter + numpy2ri.converter + pandas2ri.converter

# import R packages
base = importr("base")
tramicp = importr("tramicp")


def _run_glm_icp(
    X: np.ndarray,
    y: np.ndarray,
    envs: np.ndarray,
    alpha: float = 0.05,
) -> list[int]:
    """
    Run glmICP (logistic regression-based ICP) and return invariant feature indices.

    Parameters
    ----------
    X : (n, d) array of predictors
    y : (n,) array of binary targets (0/1)
    envs : (n,) array of environment indices
    alpha : significance level for invariance test

    Returns
    -------
    List of feature indices in the invariant set (may be empty).
    """
    n_features = X.shape[1]

    # Build data dict for R
    data_dict = {f"X{i}": X[:, i] for i in range(n_features)}
    data_dict["Y"] = y.astype(float)
    data_dict["E"] = envs

    # Formula: Y ~ X0 + X1 + ...
    predictor_names = [f"X{i}" for i in range(n_features)]
    formula_str = "Y ~ " + " + ".join(predictor_names)

    # Use converter context for numpy/pandas <-> R conversion
    with conversion.localconverter(_converter):
        # Convert to R DataFrame
        df_r = ro.DataFrame(data_dict)

        # Run glmICP
        res = tramicp.glmICP(
            formula=ro.Formula(formula_str),
            data=df_r,
            env=ro.Formula("~ E"),
            family="binomial",
            test="gcm.test",
            alpha=alpha,
            verbose=False,
        )

        # Extract the invariant set (index 0 is candidate_causal_predictors)
        candidate = res[0]
        # candidate is a numpy array like ['X0', 'X2'] or empty

    # Parse the candidate set
    if len(candidate) == 0:
        return []

    # Parse names like "X0", "X2" -> indices 0, 2
    invariant_indices = []
    for name in candidate:
        name_str = str(name)
        if name_str in predictor_names:
            invariant_indices.append(predictor_names.index(name_str))

    return invariant_indices


def _run_ranger_icp(
    X: np.ndarray,
    y: np.ndarray,
    envs: np.ndarray,
    alpha: float = 0.05,
) -> list[int]:
    """
    Run rangerICP (random forest-based ICP) and return invariant feature indices.

    Parameters
    ----------
    X : (n, d) array of predictors
    y : (n,) array of binary targets (0/1)
    envs : (n,) array of environment indices
    alpha : significance level for invariance test

    Returns
    -------
    List of feature indices in the invariant set (may be empty).
    """
    n_features = X.shape[1]

    # Build data dict for R
    data_dict = {f"X{i}": X[:, i] for i in range(n_features)}
    data_dict["Y"] = y.astype(float)
    data_dict["E"] = envs

    # Formula: Y ~ X0 + X1 + ...
    predictor_names = [f"X{i}" for i in range(n_features)]
    formula_str = "Y ~ " + " + ".join(predictor_names)

    # Use converter context for numpy/pandas <-> R conversion
    with conversion.localconverter(_converter):
        # Convert to R DataFrame
        df_r = ro.DataFrame(data_dict)

        # Run rangerICP
        res = tramicp.rangerICP(
            formula=ro.Formula(formula_str),
            data=df_r,
            env=ro.Formula("~ E"),
            test="gcm.test",
            alpha=alpha,
            verbose=False,
        )

        # Extract the invariant set (index 0 is candidate_causal_predictors)
        candidate = res[0]
        # candidate is a numpy array like ['X0', 'X2'] or empty

    # Parse the candidate set
    if len(candidate) == 0:
        return []

    # Parse names like "X0", "X2" -> indices 0, 2
    invariant_indices = []
    for name in candidate:
        name_str = str(name)
        if name_str in predictor_names:
            invariant_indices.append(predictor_names.index(name_str))

    return invariant_indices


class _BaseICPClassifier(BaseEstimator, ClassifierMixin):
    """
    Base class for ICP-based classifiers.

    Workflow:
    1. Run ICP to find the invariant set
    2. Fit a Random Forest on only those features
    3. If invariant set is empty, predict class proportions
    """

    def __init__(self, alpha: float = 0.05, n_jobs: int = 1, verbose: bool = False):
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _run_icp(self, X: np.ndarray, y: np.ndarray, envs: np.ndarray) -> list[int]:
        """Subclasses implement this to run specific ICP variant."""
        raise NotImplementedError

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        environment: np.ndarray,
    ) -> "_BaseICPClassifier":
        """
        Fit the ICP classifier.

        Parameters
        ----------
        X : (n, d) array of features
        y : (n,) array of binary labels (0/1)
        environment : (n,) array of environment indices

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        environment = np.asarray(environment).ravel()

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # Run ICP to find invariant set
        if self.verbose:
            print(f"  Running {self.__class__.__name__} with alpha={self.alpha}...")

        self.invariant_indices_ = self._run_icp(X, y, environment)

        if self.verbose:
            print(f"  Invariant set: {self.invariant_indices_}")

        if len(self.invariant_indices_) == 0:
            # Empty set: predict class proportions
            self._empty_set = True
            self._class_prior = np.mean(y)
            self._model = None
            if self.verbose:
                print(
                    f"  Empty invariant set. Will predict prior: {self._class_prior:.4f}"
                )
        else:
            # Fit RF on invariant features only
            self._empty_set = False
            X_inv = X[:, self.invariant_indices_]
            self._model = RandomForestClassifier(
                n_estimators=100, n_jobs=self.n_jobs, random_state=42
            )
            self._model.fit(X_inv, y)
            if self.verbose:
                print(
                    f"  Fitted RF on {len(self.invariant_indices_)} invariant features."
                )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Returns
        -------
        (n, 2) array with probabilities for each class.
        """
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)

        if self._empty_set:
            # Predict class prior for all samples
            n = X.shape[0]
            prob_0 = np.full(n, 1 - self._class_prior)
            prob_1 = np.full(n, self._class_prior)
            return np.column_stack([prob_0, prob_1])

        X_inv = X[:, self.invariant_indices_]
        return self._model.predict_proba(X_inv)  # type: ignore[union-attr]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return (proba[:, 1] > threshold).astype(int)


class ICPglmClassifier(_BaseICPClassifier):
    """
    ICP classifier using glmICP (logistic regression-based).

    Uses the tramicp R package to find the invariant set via
    logistic regression and GCM test.
    """

    def _run_icp(self, X: np.ndarray, y: np.ndarray, envs: np.ndarray) -> list[int]:
        return _run_glm_icp(X, y, envs, alpha=self.alpha)


class ICPrfClassifier(_BaseICPClassifier):
    """
    ICP classifier using rangerICP (random forest-based, nonparametric).

    Uses the tramicp R package to find the invariant set via
    random forests and GCM test.
    """

    def _run_icp(self, X: np.ndarray, y: np.ndarray, envs: np.ndarray) -> list[int]:
        return _run_ranger_icp(X, y, envs, alpha=self.alpha)
