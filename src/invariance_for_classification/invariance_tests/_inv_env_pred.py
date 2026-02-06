from typing import Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from ._base import InvarianceTest


def _cross_entropy_loss(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """
    Compute per-observation cross-entropy loss for multiclass classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels of shape (n_samples,), integer-encoded.
    y_prob : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).

    Returns
    -------
    losses : np.ndarray
        Per-observation cross-entropy losses of shape (n_samples,).
    """
    n_samples = len(y_true)
    # clip probabilities to avoid log(0)
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    # get the probability assigned to the true class for each observation
    losses = -np.log(y_prob[np.arange(n_samples), y_true])
    return losses


class InvariantEnvironmentPredictionTest(InvarianceTest):
    """
    Invariant Environment Prediction test for invariance.

    Tests the null hypothesis that a subset S of predictors is invariant,
    i.e., Y ⊥ E | X_S. This is done by checking if Y helps predict E
    beyond X_S:
    - Model 1: predicts E from (X_S, Y)
    - Model 2: predicts E from (X_S, permuted Y)

    If Y helps predict E (model 1 has lower cross-entropy loss), then
    Y and E are not independent given X_S, suggesting S is not invariant.

    Algorithm:
    1. Fit a classifier predicting E from (X_S, Y) and obtain OOB predictions
    2. Permute Y and fit another classifier predicting E from (X_S, permuted Y)
    3. Compute per-observation cross-entropy losses L_1 and L_2
    4. Perform one-sided paired t-test: H_0: E[L_1] = E[L_2], H_A: E[L_1] < E[L_2]
    5. Small p-value means Y helps predict E, rejecting invariance

    Parameters
    ----------
    test_classifier_type : str, default="RF"
        Classifier type to use. Supported values:
        - "RF": Random Forest (uses OOB predictions)
        - "HGBT": Histogram Gradient Boosting (uses cross-validation)
        Note: "LR" is not supported for this test.
    random_state : int, default=42
        Random state for reproducibility.
    n_folds : int, default=5
        Number of folds for cross-validation (used for HGBT).
    """

    def __init__(
        self,
        test_classifier_type: Optional[str] = None,
        random_state: int = 42,
        n_folds: int = 5,
    ):
        if test_classifier_type is None:
            test_classifier_type = "RF"

        if test_classifier_type not in ["RF", "LR", "HGBT"]:
            raise ValueError(
                f"Unknown test_classifier_type: {test_classifier_type}. "
                "Must be 'RF', 'HGBT', or 'LR'."
            )

        if test_classifier_type == "LR":
            raise NotImplementedError(
                "Logistic regression is not yet implemented for "
                "InvariantEnvironmentPredictionTest. Use 'RF' or 'HGBT' instead."
            )

        self.test_classifier_type = test_classifier_type
        self.random_state = random_state
        self.n_folds = n_folds
        self.name = "inv_env_pred"

    def _get_oob_predictions_rf(
        self, X_with_Y: np.ndarray, X_with_Y_permuted: np.ndarray, E: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get OOB predictions from random forests for environment prediction.

        Parameters
        ----------
        X_with_Y : np.ndarray
            Features including true Y, shape (n_samples, n_features + 1).
        X_with_Y_permuted : np.ndarray
            Features including permuted Y, shape (n_samples, n_features + 1).
        E : np.ndarray
            Environment labels to predict, shape (n_samples,).

        Returns
        -------
        probs_with_Y : np.ndarray
            OOB predicted probabilities for E from model with true Y.
        probs_with_Y_permuted : np.ndarray
            OOB predicted probabilities for E from model with permuted Y.
        """
        # model with true Y
        rf_with_Y = RandomForestClassifier(
            n_estimators=100, oob_score=True, random_state=self.random_state, n_jobs=1
        )
        rf_with_Y.fit(X_with_Y, E)
        probs_with_Y: np.ndarray = rf_with_Y.oob_decision_function_

        # model with permuted Y
        rf_with_Y_permuted = RandomForestClassifier(
            n_estimators=100, oob_score=True, random_state=self.random_state, n_jobs=1
        )
        rf_with_Y_permuted.fit(X_with_Y_permuted, E)
        probs_with_Y_permuted: np.ndarray = rf_with_Y_permuted.oob_decision_function_

        return probs_with_Y, probs_with_Y_permuted

    def _get_cv_predictions_hgbt(
        self, X_with_Y: np.ndarray, X_with_Y_permuted: np.ndarray, E: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cross-validated predictions from HGBT for environment prediction.

        Parameters
        ----------
        X_with_Y : np.ndarray
            Features including true Y, shape (n_samples, n_features + 1).
        X_with_Y_permuted : np.ndarray
            Features including permuted Y, shape (n_samples, n_features + 1).
        E : np.ndarray
            Environment labels to predict, shape (n_samples,).

        Returns
        -------
        probs_with_Y : np.ndarray
            CV predicted probabilities for E from model with true Y.
        probs_with_Y_permuted : np.ndarray
            CV predicted probabilities for E from model with permuted Y.
        """
        n_samples = len(E)
        n_classes = len(np.unique(E))
        probs_with_Y = np.zeros((n_samples, n_classes))
        probs_with_Y_permuted = np.zeros((n_samples, n_classes))

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in kf.split(X_with_Y):
            # model with true Y
            hgbt_with_Y = HistGradientBoostingClassifier(random_state=self.random_state)
            hgbt_with_Y.fit(X_with_Y[train_idx], E[train_idx])
            probs_with_Y[test_idx] = hgbt_with_Y.predict_proba(X_with_Y[test_idx])

            # model with permuted Y
            hgbt_with_Y_permuted = HistGradientBoostingClassifier(
                random_state=self.random_state
            )
            hgbt_with_Y_permuted.fit(X_with_Y_permuted[train_idx], E[train_idx])
            probs_with_Y_permuted[test_idx] = hgbt_with_Y_permuted.predict_proba(
                X_with_Y_permuted[test_idx]
            )

        return probs_with_Y, probs_with_Y_permuted

    def test(self, X: np.ndarray, y: np.ndarray, E: np.ndarray) -> float:
        """
        Perform the Invariant Environment Prediction test.

        Parameters
        ----------
        X : np.ndarray
            Predictor subset X_S of shape (n_samples, n_features).
            Can be empty (n_features=0) to test the empty set.
        y : np.ndarray
            Target variable of shape (n_samples,).
        E : np.ndarray
            Environment labels of shape (n_samples,).

        Returns
        -------
        p_value : float
            p-value of the one-sided t-test. Small values indicate that Y
            helps predict E given X_S, suggesting S is not invariant.
        """
        if len(np.unique(E)) < 2:
            # test undefined for single environment; assume invariance
            return 1.0

        # encode environment as numeric for the target
        le_E = LabelEncoder()
        E_encoded: np.ndarray = np.asarray(le_E.fit_transform(E))

        # handle X dimensions
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # reshape y if needed
        y_col = np.asarray(y).reshape(-1, 1)

        # create feature matrices
        if X.shape[1] == 0:
            X_with_Y = y_col
        else:
            X_with_Y = np.hstack([X, y_col])

        # permute Y for the second model
        rng = np.random.RandomState(self.random_state)
        y_permuted = rng.permutation(y).reshape(-1, 1)

        if X.shape[1] == 0:
            X_with_Y_permuted = y_permuted
        else:
            X_with_Y_permuted = np.hstack([X, y_permuted])

        # get OOB/CV predictions
        probs_with_Y: np.ndarray = np.array([])
        probs_with_Y_permuted: np.ndarray = np.array([])
        try:
            if self.test_classifier_type == "RF":
                probs_with_Y, probs_with_Y_permuted = self._get_oob_predictions_rf(
                    X_with_Y, X_with_Y_permuted, E_encoded
                )
            elif self.test_classifier_type == "HGBT":
                probs_with_Y, probs_with_Y_permuted = self._get_cv_predictions_hgbt(
                    X_with_Y, X_with_Y_permuted, E_encoded
                )
        except Exception:
            # if fitting fails, return 1.0 (cannot reject invariance)
            return 1.0

        # handle NaN predictions (can occur if sample never in OOB)
        if np.any(np.isnan(probs_with_Y)) or np.any(np.isnan(probs_with_Y_permuted)):
            return 1.0

        # compute per-observation cross-entropy losses
        L_1 = _cross_entropy_loss(E_encoded, probs_with_Y)
        L_2 = _cross_entropy_loss(E_encoded, probs_with_Y_permuted)

        # one-sided paired t-test
        # H_0: E[L_1] = E[L_2]
        # H_A: E[L_1] < E[L_2] (Y helps predict E, so model 1 has lower loss)
        result = stats.ttest_rel(L_1, L_2, alternative="less")
        p_value = float(result.pvalue)

        if np.isnan(p_value):
            return 1.0

        return p_value
