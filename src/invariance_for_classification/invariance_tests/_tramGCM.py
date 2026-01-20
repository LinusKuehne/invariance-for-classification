import numpy as np
import pandas as pd

from ._base import InvarianceTest

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
except ImportError:
    ro = None
    pandas2ri = None
    localconverter = None
    importr = None


class TramGcmTest(InvarianceTest):
    """
    Tram-GCM test

    Tests the null hypothesis that the invariant set is the given set of predictors.
    Uses the R package 'tramicp'.

    Parameters
    ----------
    test_classifier_type : str, default="RF"
        "RF" for random forest (rangerICP), "LR" for logistic regression (glmICP).
    """

    def __init__(
        self,
        test_classifier_type: str = "RF",
    ):
        if ro is None or importr is None:
            raise ImportError("rpy2 is required for TramGcmTest")

        self.test_classifier_type = test_classifier_type
        self.name = "tram_gcm"

        try:
            self.tramicp = importr("tramicp")
        except Exception as e:
            raise ImportError(
                f"R package 'tramicp' not found or could not be loaded: {e}"
            ) from e

    def test(self, X: np.ndarray, y: np.ndarray, E: np.ndarray) -> float:
        """
        Perform the Tram-GCM test.

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
        if len(np.unique(E)) < 2:
            return 1.0

        # Assert rpy2 modules are available (init would have failed otherwise)
        assert ro is not None
        assert pandas2ri is not None
        assert localconverter is not None

        _, n_features = X.shape

        # construct DataFrame
        # assign generic names X0, X1, ... to the predictors
        feature_names = [f"X{i}" for i in range(n_features)]

        # create a dict first to avoid fragmentation
        data_dict = {name: X[:, i] for i, name in enumerate(feature_names)}
        data_dict["Y"] = y

        # convert E to string first to ensure it's treated as categorical/factor in R
        data_dict["Env"] = E.astype(str)

        df = pd.DataFrame(data_dict)

        # explicitly mark Env as category for R factor conversion
        df["Env"] = df["Env"].astype("category")

        # select formula and target set name
        if n_features == 0:
            formula_str = "Y ~ 1"
            target_set_name = "Empty"
        else:
            # join with '+' with no spaces, as per R code example
            rhs = "+".join(feature_names)
            formula_str = f"Y ~ {rhs}"
            target_set_name = rhs

        # convert to R DataFrame
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)

        r_formula = ro.Formula(formula_str)

        # the environment variable is "Env" in the dataframe
        r_env_formula = ro.Formula("~ Env")

        # Run the appropriate ICP function
        try:
            if self.test_classifier_type == "RF":
                res = self.tramicp.rangerICP(
                    formula=r_formula,
                    data=r_df,
                    env=r_env_formula,
                    test="gcm.test",
                    verbose=False,
                )
            elif self.test_classifier_type == "LR":
                res = self.tramicp.glmICP(
                    formula=r_formula,
                    data=r_df,
                    env=r_env_formula,
                    family="binomial",
                    verbose=False,
                )
            else:
                raise ValueError(
                    f"Unknown test_classifier_type: {self.test_classifier_type}"
                )

            # Extract p-values
            # returns a named vector
            pvals_vec = self.tramicp.pvalues(res, "set")
            pvals_dict = dict(zip(pvals_vec.names, pvals_vec, strict=True))

            # Look up the p-value for the tested set
            # We try exact match first
            p_value = pvals_dict.get(target_set_name)

            if p_value is None:
                # Fallback: try sorted keys if order differs
                # (e.g. if tramicp reorders X1+X0 to X0+X1)
                sorted_rhs = "+".join(sorted(feature_names))
                p_value = pvals_dict.get(sorted_rhs)

            if p_value is None:
                # If still not found, this is unexpected for the full set test.
                # We might be in a situation where variables were dropped?
                # Raise error to debug
                raise RuntimeError(
                    f"Could not find p-value for set '{target_set_name}' in tramicp results. "
                    f"Available keys: {list(pvals_dict.keys())}"
                )

            # Handle R's NA (which comes as NaN in python float)
            if np.isnan(p_value):
                # As per provided R code, sample randomly if NA
                return np.random.uniform(0.0, 1.0)

            return float(p_value)

        except Exception as e:
            # If it's the specific RuntimeError above, re-raise
            if "Could not find p-value" in str(e):
                raise
            raise RuntimeError(f"tramicp execution failed: {e}") from e
