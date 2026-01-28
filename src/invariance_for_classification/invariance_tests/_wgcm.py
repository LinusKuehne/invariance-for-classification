"""
Weighted Generalised Covariance Measure (WGCM) Invariance Test.

This implements the WGCM test for conditional independence,
adapted for invariance testing in classification settings.

Based on:
Cyrill Scheidegger, Julia Hoerrmann, Peter Buehlmann:
"The Weighted Generalised Covariance Measure"
http://jmlr.org/papers/v23/21-1328.html
"""

from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

from ._base import InvarianceTest


class WGCMTest(InvarianceTest):
    """
    Weighted Generalized Covariance Measure (WGCM) Test for Invariance.

    Tests conditional independence of environment E and target Y given predictors X_S.
    In the notation of the original R package: X=E (environment), Y=y (target), Z=X_S (predictors).

    H0: E ⊥ Y | X_S  (which implies invariance of Y|X_S across environments)

    The test:
    1. Splits data into train/test sets
    2. Estimates residuals eps = E - E[E|X_S] and xi = Y - E[Y|X_S]
    3. Estimates a weight function W = sign(E[eps*xi|X_S]) on training data
    4. Computes test statistic on test data using the estimated weights

    Parameters
    ----------
    beta : float, default=0.3
        Fraction of data used for estimating the weight function (training).
    max_nrounds : int, default=500
        Maximum number of boosting rounds for xgboost.
    eta : list of float, default=[0.1, 0.2, 0.3, 0.5]
        Learning rates to try in CV.
    max_depth : list of int, default=[1, 2, 3, 4, 5, 6, 7]
        Max depths to try in CV.
    early_stopping_rounds : int, default=10
        Early stopping rounds for xgboost CV.
    k_cv : int, default=10
        Number of CV folds for hyperparameter tuning.
    use_categorical_loss : bool, default=False
        If True, use logistic loss for binary Y and multi:softprob for
        multi-class E instead of squared error. This may improve probability
        estimates for categorical variables.
    test_classifier_type: Optional[str], default=None
        not implemented, included for compatibility.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        beta: float = 0.3,
        max_nrounds: int = 500,
        eta: Optional[list] = None,
        max_depth: Optional[list] = None,
        early_stopping_rounds: int = 10,
        k_cv: int = 10,
        use_categorical_loss: bool = False,
        test_classifier_type: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        # Validate test_classifier_type for compatibility with other tests
        # WGCMTest only uses xgboost internally, but we accept RF/LR for API compatibility
        valid_types = [None, "RF", "LR", "xgb"]
        if test_classifier_type not in valid_types:
            raise ValueError(
                f"Unknown test_classifier_type: {test_classifier_type}. "
                f"Valid options are: {valid_types}"
            )

        self.beta = beta
        self.max_nrounds = max_nrounds
        self.eta = eta if eta is not None else [0.1, 0.2, 0.3, 0.5]
        self.max_depth = max_depth if max_depth is not None else [1, 2, 3, 4, 5, 6, 7]
        self.early_stopping_rounds = early_stopping_rounds
        self.k_cv = k_cv
        self.use_categorical_loss = use_categorical_loss
        self.test_classifier_type = (
            test_classifier_type  # stored but not used (xgboost only)
        )
        # Default random_state to 42 for reproducibility if not specified
        self.random_state = random_state if random_state is not None else 42
        self.name = "wgcm"

    def test(self, X: np.ndarray, y: np.ndarray, E: np.ndarray) -> float:
        """
        Perform the WGCM test for invariance.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Subset of predictors to test (Z in R notation).
        y : np.ndarray of shape (n_samples,)
            Target variable (Y in R notation), binary {0, 1}.
        E : np.ndarray of shape (n_samples,)
            Environment indicator (X in R notation), will be converted to numeric.

        Returns
        -------
        p_value : float
            The p-value of the test. High p-values indicate that null hypothesis
            (conditional independence / invariance) cannot be rejected.
        """
        n = len(y)

        # Convert E to numeric if needed (one-hot or label encoding)
        E_numeric = self._encode_environment(E)

        # Handle case with no conditioning variables
        if X is None or X.shape[1] == 0:
            return self._test_unconditional(E_numeric, y)

        # Ensure X is 2D
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and X.shape[1] != n:
            X = X.T

        # Check if E and Y are both 1D (original wgcm.est case)
        if E_numeric.ndim == 1 and y.ndim == 1:
            return self._wgcm_est_1d(E_numeric, y, X)
        else:
            # Multivariate case
            return self._wgcm_est_mult(E_numeric, y, X)

    def _encode_environment(self, E: np.ndarray) -> np.ndarray:
        """Convert environment labels to numeric values."""
        E = np.asarray(E)
        if E.ndim == 1:
            # Check if already numeric
            if np.issubdtype(E.dtype, np.floating):
                return E.astype(float)
            # Label encode
            unique_envs = np.unique(E)
            env_map = {env: i for i, env in enumerate(unique_envs)}
            return np.array([env_map[e] for e in E], dtype=float)
        return E.astype(float)

    def _test_unconditional(self, E: np.ndarray, y: np.ndarray) -> float:
        """Test without conditioning variables (simple correlation test)."""
        n = len(y)
        eps = E - np.mean(E)
        xi = y - np.mean(y)
        R = eps * xi
        var_R = np.mean(R**2) - np.mean(R) ** 2
        if var_R <= 0:
            return 1.0
        T_stat = np.sqrt(n) * np.mean(R) / np.sqrt(var_R)
        p_value = 2 * stats.norm.cdf(-np.abs(T_stat))
        return float(p_value)

    def _wgcm_est_1d(self, E: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
        """
        WGCM test with estimated weight function for 1D E and Y.

        Parameters
        ----------
        E : np.ndarray of shape (n,)
            Environment indicator (numeric).
        y : np.ndarray of shape (n,)
            Target variable.
        Z : np.ndarray of shape (n, d)
            Conditioning variables.
        """
        n = len(y)
        rng = np.random.default_rng(self.random_state)

        # Split into train/test
        n_train = int(np.ceil(self.beta * n))
        indices = rng.permutation(n)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        Z_train, Z_test = Z[train_idx], Z[test_idx]
        E_train, E_test = E[train_idx], E[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Estimate weight function on training data
        W = self._predict_weight(E_train, y_train, Z_train, Z_test)

        # Compute test statistic on test data
        p_value = self._wgcm_1d_1sided(E_test, y_test, Z_test, W)

        return p_value

    def _wgcm_est_mult(
        self, E: np.ndarray, y: np.ndarray, Z: np.ndarray, nsim: int = 499
    ) -> float:
        """
        WGCM test with estimated weight function for multivariate case.
        """
        # Ensure 2D
        if E.ndim == 1:
            E = E.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n = E.shape[0]
        d_E = E.shape[1]
        d_y = y.shape[1]

        rng = np.random.default_rng(self.random_state)

        # Split into train/test
        n_train = int(np.ceil(self.beta * n))
        indices = rng.permutation(n)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        Z_train, Z_test = Z[train_idx], Z[test_idx]
        E_train, E_test = E[train_idx], E[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        n_test = len(test_idx)

        # Compute residuals on test data (E and Y are categorical)
        eps_mat = np.column_stack(
            [
                self._get_residuals(E_test[:, j], Z_test, is_categorical=True)
                for j in range(d_E)
            ]
        )
        xi_mat = np.column_stack(
            [
                self._get_residuals(y_test[:, col_idx], Z_test, is_categorical=True)
                for col_idx in range(d_y)
            ]
        )

        # Build R matrix
        R_list = []
        for j in range(d_E):
            for col_idx in range(d_y):
                W = self._predict_weight(
                    E_train[:, j], y_train[:, col_idx], Z_train, Z_test
                )
                R_jl = eps_mat[:, j] * xi_mat[:, col_idx] * W
                R_list.append(R_jl)

        R = np.array(R_list)  # shape (d_E * d_y, n_test)

        # Normalize
        R_var = np.mean(R**2, axis=1) - np.mean(R, axis=1) ** 2
        R_var = np.maximum(R_var, 1e-10)  # Avoid division by zero
        R_norm = R / np.sqrt(R_var)[:, np.newaxis]

        # One-sided test (estimated weights aim for positive statistic)
        T_stat = np.sqrt(n_test) * np.max(np.mean(R_norm, axis=1))

        # Simulate null distribution
        sim_matrix = rng.standard_normal((n_test, nsim))
        T_stat_sim = np.max(R_norm @ sim_matrix, axis=0) / np.sqrt(n_test)

        p_value = (np.sum(T_stat_sim >= T_stat) + 1) / (nsim + 1)

        return float(p_value)

    def _fit_xgboost(
        self, Y: np.ndarray, X: np.ndarray, is_categorical: bool = False
    ) -> xgb.Booster:
        """
        Fit xgboost model with cross-validation for hyperparameter selection.

        Parameters
        ----------
        Y : np.ndarray of shape (n,)
            Target variable.
        X : np.ndarray of shape (n, d)
            Features.
        is_categorical : bool, default=False
            If True and use_categorical_loss is enabled, use appropriate
            classification objective (logistic for binary, softprob for multiclass).

        Returns
        -------
        model : xgb.Booster
            Fitted xgboost model.
        """
        # Determine objective and eval metric
        objective = "reg:squarederror"
        eval_metric = "rmse"
        num_class = None

        if is_categorical and self.use_categorical_loss:
            unique_vals = np.unique(Y)
            n_classes = len(unique_vals)
            if n_classes == 2:
                # Binary classification
                objective = "binary:logistic"
                eval_metric = "logloss"
            elif n_classes > 2:
                # Multi-class classification
                objective = "multi:softprob"
                eval_metric = "mlogloss"
                num_class = n_classes
                # Ensure labels are 0, 1, 2, ... for multi-class
                label_map = {v: i for i, v in enumerate(unique_vals)}
                Y = np.array([label_map[y] for y in Y])

        dtrain = xgb.DMatrix(X, label=Y, nthread=1)

        best_score = np.inf
        best_params = {"eta": self.eta[0], "max_depth": self.max_depth[0]}
        best_nrounds = 1

        # Grid search over hyperparameters
        for eta_val in self.eta:
            for depth_val in self.max_depth:
                params = {
                    "eta": eta_val,
                    "max_depth": depth_val,
                    "objective": objective,
                    "eval_metric": eval_metric,
                    "nthread": 1,
                    "verbosity": 0,
                }
                if num_class is not None:
                    params["num_class"] = num_class

                try:
                    cv_result = xgb.cv(
                        params,
                        dtrain,
                        num_boost_round=self.max_nrounds,
                        nfold=min(self.k_cv, len(Y)),
                        early_stopping_rounds=self.early_stopping_rounds,
                        verbose_eval=False,
                    )

                    # Get the metric column name
                    metric_col = f"test-{eval_metric}-mean"
                    # cv_result is a DataFrame; extract Series and compute
                    metric_values: pd.Series = cv_result[metric_col]  # type: ignore[assignment]
                    min_score = float(metric_values.min())
                    opt_rounds = int(metric_values.idxmin()) + 1

                    if min_score < best_score:
                        best_score = min_score
                        best_params = {"eta": eta_val, "max_depth": depth_val}
                        best_nrounds = opt_rounds
                except Exception:
                    continue

        # Fit final model with best parameters
        final_params = {
            "eta": best_params["eta"],
            "max_depth": best_params["max_depth"],
            "objective": objective,
            "nthread": 1,
            "verbosity": 0,
        }
        if num_class is not None:
            final_params["num_class"] = num_class

        model = xgb.train(final_params, dtrain, num_boost_round=best_nrounds)

        return model

    def _get_residuals(
        self, Y: np.ndarray, X: np.ndarray, is_categorical: bool = False
    ) -> np.ndarray:
        """
        Get residuals Y - E[Y|X] using xgboost.

        Parameters
        ----------
        Y : np.ndarray of shape (n,)
            Target variable.
        X : np.ndarray of shape (n, d)
            Features.
        is_categorical : bool, default=False
            If True and use_categorical_loss is enabled, use classification objective.

        Returns
        -------
        residuals : np.ndarray of shape (n,)
            Residuals Y - predictions.
        """
        # Store original Y for residual computation
        Y_original = Y.copy()

        # For multi-class with categorical loss, we need to handle label mapping
        unique_vals = np.unique(Y)
        n_classes = len(unique_vals)
        label_map = None

        if is_categorical and self.use_categorical_loss and n_classes > 2:
            label_map = {v: i for i, v in enumerate(unique_vals)}
            Y_mapped = np.array([label_map[y] for y in Y])
        else:
            Y_mapped = Y

        model = self._fit_xgboost(Y_mapped, X, is_categorical=is_categorical)
        dmat = xgb.DMatrix(X, nthread=1)
        predictions = model.predict(dmat)

        # For multi-class softprob, predictions are (n, num_classes)
        # We need E[Y|X] which for integer-encoded classes is sum(k * P(Y=k|X))
        if is_categorical and self.use_categorical_loss and n_classes > 2:
            # predictions shape: (n, num_classes)
            # Compute expected value: sum over k of k * P(Y=k|X)
            # But we want residuals w.r.t. original encoding
            assert label_map is not None  # guaranteed when n_classes > 2
            inv_label_map = {i: v for v, i in label_map.items()}
            class_values = np.array([inv_label_map[i] for i in range(n_classes)])
            predictions = predictions @ class_values  # E[Y|X]

        return Y_original - predictions

    def _predict_weight(
        self,
        E_train: np.ndarray,
        y_train: np.ndarray,
        Z_train: np.ndarray,
        Z_test: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate weight function W = sign(E[eps*xi|Z]) on test data.

        Parameters
        ----------
        E_train : np.ndarray of shape (n_train,)
            Environment indicator on training data.
        y_train : np.ndarray of shape (n_train,)
            Target on training data.
        Z_train : np.ndarray of shape (n_train, d)
            Conditioning variables on training data.
        Z_test : np.ndarray of shape (n_test, d)
            Conditioning variables on test data.

        Returns
        -------
        W : np.ndarray of shape (n_test,)
            Predicted weights for test data.
        """
        # Get residuals on training data
        # E is categorical (environment indicator), Y is categorical (binary target)
        eps = self._get_residuals(E_train, Z_train, is_categorical=True)
        xi = self._get_residuals(y_train, Z_train, is_categorical=True)

        # Product of residuals
        eps_xi = eps * xi

        # Fit model to predict eps*xi from Z (this is regression, not categorical)
        model = self._fit_xgboost(eps_xi, Z_train, is_categorical=False)

        # Predict on test data
        dtest = xgb.DMatrix(Z_test, nthread=1)
        W = model.predict(dtest)

        return W

    def _wgcm_1d_1sided(
        self,
        E_test: np.ndarray,
        y_test: np.ndarray,
        Z_test: np.ndarray,
        W: np.ndarray,
    ) -> float:
        """
        Compute one-sided p-value for WGCM with given weights.

        The test statistic is expected to be positive under the alternative,
        so we use a one-sided test.

        Parameters
        ----------
        E_test : np.ndarray of shape (n_test,)
            Environment indicator on test data.
        y_test : np.ndarray of shape (n_test,)
            Target on test data.
        Z_test : np.ndarray of shape (n_test, d)
            Conditioning variables on test data.
        W : np.ndarray of shape (n_test,)
            Weight function values.

        Returns
        -------
        p_value : float
            One-sided p-value.
        """
        n = len(y_test)

        # Get residuals on test data (both E and Y are categorical)
        eps = self._get_residuals(E_test, Z_test, is_categorical=True)
        xi = self._get_residuals(y_test, Z_test, is_categorical=True)

        # Weighted product of residuals
        R = eps * xi * W

        # Test statistic
        var_R = np.mean(R**2) - np.mean(R) ** 2
        if var_R <= 0:
            return 1.0

        T_stat = np.sqrt(n) * np.mean(R) / np.sqrt(var_R)

        # One-sided p-value (expected positive under alternative)
        p_value = 1 - stats.norm.cdf(T_stat)

        return float(p_value)
