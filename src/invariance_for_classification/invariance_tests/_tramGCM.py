"""Code for Tram-GCM test (Python implementation of parts of https://github.com/LucasKook/tramicp)."""

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from ._base import InvarianceTest


class TramGcmTest(InvarianceTest):
    """
    Tram-GCM test (Python implementation)

    Tests the null hypothesis that the given set of predictors is invariant.
    Re-implementation of the TRAM-GCM test from the R package 'tramicp' for binary classification in Python.

    Parameters
    ----------
    test_classifier_type : str, default="RF"
        "RF" for random forest (rangerICP equivalent),
        "LR" for logistic regression (glmICP equivalent),
        "HGBT" for histogram gradient boosting.

    Notes
    -----
    For RF mode: tramicp's rangerICP passes Y as numeric (not factor) to R's ranger,
    so it uses regression mode. We replicate this by using RandomForestRegressor.
    The residuals are Y - prediction where prediction approximates E[Y|X].

    For LR mode: tramicp's glmICP uses binomial GLM with response residuals (Y - fitted_prob).
    We use LogisticRegression to get fitted probabilities.

    For HGBT mode: Uses HistGradientBoostingRegressor for regression.
    """

    def __init__(
        self,
        test_classifier_type: str = "RF",
    ):
        if test_classifier_type not in ["RF", "LR", "HGBT"]:
            raise ValueError(
                f"Unknown test_classifier_type: {test_classifier_type}. "
                "Must be 'RF', 'LR', or 'HGBT'."
            )
        self.test_classifier_type = test_classifier_type
        self.name = "tram_gcm"

    def test(self, X: np.ndarray, y: np.ndarray, E: np.ndarray) -> float:
        """
        Perform the TRAM-GCM test.

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
            p-value of Tram-GCM test
        """
        # if only one environment, we cannot test invariance across environments
        if len(np.unique(E)) < 2:
            return 1.0

        n_samples, n_features = X.shape

        # --- Step 1: Compute residuals for Y ~ X ---

        # For RF: tramicp uses regression forest on numeric Y (not probability forest)
        # For LR: tramicp uses binomial GLM with response residuals
        # For HGBT: uses histogram gradient boosting regressor
        if n_features == 0:
            # null model: predict global mean everywhere
            mean_y = np.mean(y)
            y_hat = np.full_like(y, mean_y, dtype=float)
        else:
            if self.test_classifier_type == "RF":
                # tramicp's RANGER uses regression when Y is numeric (not a factor)
                # Ranger defaults: mtry = floor(sqrt(p)), min.node.size = 5 for regression
                mtry = max(1, int(np.sqrt(n_features)))
                reg = RandomForestRegressor(
                    n_estimators=500,
                    max_features=mtry,
                    min_samples_leaf=5,
                    random_state=42,
                    bootstrap=True,
                    n_jobs=1,
                )
                reg.fit(X, y)
                y_hat = reg.predict(X)
            elif self.test_classifier_type == "HGBT":
                reg = HistGradientBoostingRegressor(random_state=42)
                reg.fit(X, y)
                y_hat = reg.predict(X)
            elif self.test_classifier_type == "LR":
                # 'tramicp' uses glm with family="binomial", which has no penalty by default
                # in sklearn, LogisticRegression defaults to L2 penalty
                # => use a large C (inverse regularization strength) to approximate unpenalized regression
                clf = LogisticRegression(
                    C=1e10, solver="lbfgs", max_iter=1000, random_state=42
                )
                clf.fit(X, y)
                y_hat = clf.predict_proba(X)[:, 1]
            else:
                raise ValueError(
                    f"Unknown test_classifier_type: {self.test_classifier_type}"
                )

        # residuals r_y = Y - E[Y|X]
        r_y = y - y_hat

        # --- Step 2: Compute residuals for E ~ X ---

        # need to test independence between r_y and E given X
        # done by checking correlation between r_y and residuals of E ~ X

        # prepare environment matrix
        # 'tramicp' converts Env to model matrix (dummies) and drops intercept
        if E.ndim == 1:
            E_reshaped = E.reshape(-1, 1)
            # one-hot encode, dropping first category to avoid perfect multicollinearity (k-1 dummies)
            # matches R's model.matrix(~E) behavior after removing intercept
            enc = OneHotEncoder(drop="first", sparse_output=False)
            E_mat = enc.fit_transform(E_reshaped)
        else:
            E_mat = E

        # if E is constant (e.g. only 1 level after filtering), E_mat might be empty
        if E_mat.shape[1] == 0:
            return 1.0

        # regress each column of E on X
        r_e_list = []
        for i in range(E_mat.shape[1]):
            e_col = E_mat[:, i]
            if n_features == 0:
                e_hat = np.mean(e_col)
                res_e = e_col - e_hat
            else:
                if self.test_classifier_type == "HGBT":
                    reg_e = HistGradientBoostingRegressor(random_state=42)
                else:
                    # 'tramicp' uses RF for E ~ X (gcm.test default)
                    # even if E columns are binary dummies, they are treated as numeric targets in 'ranger'
                    # Ranger defaults: mtry = floor(sqrt(p)), min.node.size = 5 for regression
                    mtry = max(1, int(np.sqrt(n_features)))
                    reg_e = RandomForestRegressor(
                        n_estimators=500,
                        max_features=mtry,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=1,
                    )
                reg_e.fit(X, e_col)
                e_pred = reg_e.predict(X)
                res_e = e_col - e_pred
            r_e_list.append(res_e)

        r_e = np.column_stack(r_e_list)

        # --- Step 3: GCM test statistic ---

        # R_mat[i, j] = r_y[i] * r_e[i, j]
        # (broadcasting r_y across columns of r_e)
        R_mat = r_e * r_y[:, np.newaxis]

        # estimate cov matrix of R
        # => use sample covariance (dividing by n) as in 'tramicp' code
        R_mean = np.mean(R_mat, axis=0)
        R_centered = R_mat - R_mean
        Sigma = (R_centered.T @ R_centered) / n_samples

        # compute Sigma^(-1/2) using eigendecomp
        eigvals, eigvecs = np.linalg.eigh(Sigma)

        # filter small eigenvalues to handle singularity
        tol = 1e-12
        mask = eigvals > tol

        if not np.any(mask):
            # if variance is effectively 0: can't compute statistic
            # implies residuals are visibly 0 or constant
            return 1.0

        inv_sqrt_eigvals = np.zeros_like(eigvals)
        inv_sqrt_eigvals[mask] = 1.0 / np.sqrt(eigvals[mask])

        # construct Sigma^(-1/2)
        SigInvHalf = eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T

        # T_stat = Sigma^(-1/2) * (1/sqrt(n)) * sum(R)
        sum_R = np.sum(R_mat, axis=0)
        t_stat_vec = SigInvHalf @ sum_R / np.sqrt(n_samples)

        # final test statistic (squared norm)
        test_stat = np.sum(t_stat_vec**2)

        # p-value from Chi-squared distr
        # dof is dimension of E (number of columns in E_mat)
        dE = E_mat.shape[1]

        p_value = stats.chi2.sf(test_stat, df=dE)

        if np.isnan(p_value):
            return 1.0

        return float(p_value)
