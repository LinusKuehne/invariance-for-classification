import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from ._base import InvarianceTest


def _delong_roc_variance(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """
    Compute AUC and its variance using the DeLong method.

    Based on: DeLong et al. (1988) and the fast algorithm by Sun & Xu (2014).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_samples,)
    y_score : np.ndarray
        Predicted scores of shape (n_samples,) or (n_samples, n_models)

    Returns
    -------
    auc : np.ndarray
        AUC value(s)
    var : np.ndarray
        Variance-covariance matrix of AUC(s)
    """
    y_true = np.asarray(y_true)
    y_score = np.atleast_2d(y_score).T if y_score.ndim == 1 else y_score

    pos_mask = y_true == 1
    neg_mask = ~pos_mask
    n_pos, n_neg = pos_mask.sum(), neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        n_models = y_score.shape[1]
        return np.full(n_models, 0.5), np.eye(n_models) * 0.25

    scores_pos = y_score[pos_mask]  # (n_pos, n_models)
    scores_neg = y_score[neg_mask]  # (n_neg, n_models)

    # compute AUC and placement values using vectorized operations
    # V10[i,m] = P(score_neg < score_pos[i]) for model m
    # V01[j,m] = P(score_pos > score_neg[j]) for model m
    V10 = np.mean(
        (scores_pos[:, np.newaxis, :] > scores_neg[np.newaxis, :, :]).astype(float)
        + 0.5 * (scores_pos[:, np.newaxis, :] == scores_neg[np.newaxis, :, :]),
        axis=1,
    )  # (n_pos, n_models)

    V01 = np.mean(
        (scores_pos[np.newaxis, :, :] > scores_neg[:, np.newaxis, :]).astype(float)
        + 0.5 * (scores_pos[np.newaxis, :, :] == scores_neg[:, np.newaxis, :]),
        axis=1,
    )  # (n_neg, n_models)

    auc = V10.mean(axis=0)

    # variance-covariance matrix
    S10 = np.cov(V10, rowvar=False) if n_pos > 1 else np.zeros((V10.shape[1],) * 2)
    S01 = np.cov(V01, rowvar=False) if n_neg > 1 else np.zeros((V01.shape[1],) * 2)

    # handle single model case
    if y_score.shape[1] == 1:
        S10 = np.atleast_2d(S10)
        S01 = np.atleast_2d(S01)

    var = S10 / n_pos + S01 / n_neg
    return auc, var


def delong_test(
    y_true: np.ndarray,
    y_score1: np.ndarray,
    y_score2: np.ndarray,
    alternative: str = "less",
) -> float:
    """
    Perform DeLong's test comparing two correlated ROC curves.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_score1 : np.ndarray
        Predicted probabilities from model 1 (without E)
    y_score2 : np.ndarray
        Predicted probabilities from model 2 (with E)
    alternative : str
        "less": test if AUC1 < AUC2, "greater": AUC1 > AUC2, "two-sided": AUC1 != AUC2

    Returns
    -------
    p_value : float
        p-value of the test
    """
    y_scores = np.column_stack([y_score1, y_score2])
    aucs, var = _delong_roc_variance(y_true, y_scores)

    # variance of AUC1 - AUC2
    var_diff = var[0, 0] + var[1, 1] - 2 * var[0, 1]
    if var_diff <= 0:
        return 1.0

    z = (aucs[0] - aucs[1]) / np.sqrt(var_diff)

    if alternative == "less":
        return float(stats.norm.cdf(z))
    elif alternative == "greater":
        return float(1 - stats.norm.cdf(z))
    return float(2 * stats.norm.cdf(-abs(z)))


class DeLongTest(InvarianceTest):
    """
    DeLong test for invariance.

    Tests the null hypothesis that a subset S of predictors is invariant,
    i.e., Y ⊥ E | X_S. This is done by comparing the predictive performance
    of two models:
    - Model without E: predicts Y from X_S (with permuted E)
    - Model with E: predicts Y from (X_S, E)

    If including E significantly improves prediction (higher AUC), then S is
    not invariant. DeLong's test is used to compare the ROC curves.

    Algorithm:
    1. Fit a classifier predicting Y from (X_S, E) and obtain predictions
    2. Permute E and fit another classifier (effectively without E information)
    3. Compare ROC curves using DeLong's test with alternative="less"
    4. Small p-value means E helps prediction, rejecting invariance

    Parameters
    ----------
    test_classifier_type : str, default="RF"
        "RF" for random forest, "LR" for logistic regression.
    n_folds : int, default=10
        Number of folds for cross-validation (used for LR to get predictions).
    random_state : int, default=42
        Random state for reproducibility.
    """

    def __init__(
        self,
        test_classifier_type: str = "RF",
        n_folds: int = 10,
        random_state: int = 42,
    ):
        self.test_classifier_type = test_classifier_type
        self.n_folds = n_folds
        self.random_state = random_state

        if test_classifier_type not in ["RF", "LR", "HGBT"]:
            raise ValueError(f"Unknown test_classifier_type: {test_classifier_type}")

        self.name = "delong"

    def _get_predictions_rf(
        self, X_with_E: np.ndarray, X_without_E: np.ndarray, y: np.ndarray
    ) -> tuple:
        """Get OOB predictions from random forests."""
        # model with E
        rf_with_E = RandomForestClassifier(
            n_estimators=100, oob_score=True, random_state=self.random_state, n_jobs=1
        )
        rf_with_E.fit(X_with_E, y)
        preds_with_E = rf_with_E.oob_decision_function_[:, 1]

        # model without E (permuted E)
        rf_without_E = RandomForestClassifier(
            n_estimators=100, oob_score=True, random_state=self.random_state, n_jobs=1
        )
        rf_without_E.fit(X_without_E, y)
        preds_without_E = rf_without_E.oob_decision_function_[:, 1]

        return preds_with_E, preds_without_E

    def _get_predictions_lr(
        self, X_with_E: np.ndarray, X_without_E: np.ndarray, y: np.ndarray
    ) -> tuple:
        """Get cross-validated predictions from logistic regression."""
        n_samples = len(y)
        preds_with_E = np.zeros(n_samples)
        preds_without_E = np.zeros(n_samples)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in kf.split(X_with_E):
            # model with E
            lr_with_E = LogisticRegression(
                random_state=self.random_state, max_iter=1000
            )
            lr_with_E.fit(X_with_E[train_idx], y[train_idx])
            preds_with_E[test_idx] = lr_with_E.predict_proba(X_with_E[test_idx])[:, 1]

            # model without E (permuted E)
            lr_without_E = LogisticRegression(
                random_state=self.random_state, max_iter=1000
            )
            lr_without_E.fit(X_without_E[train_idx], y[train_idx])
            preds_without_E[test_idx] = lr_without_E.predict_proba(
                X_without_E[test_idx]
            )[:, 1]

        return preds_with_E, preds_without_E

    def _get_predictions_hgbt(
        self, X_with_E: np.ndarray, X_without_E: np.ndarray, y: np.ndarray
    ) -> tuple:
        """Get cross-validated predictions from histogram gradient boosting."""
        n_samples = len(y)
        preds_with_E = np.zeros(n_samples)
        preds_without_E = np.zeros(n_samples)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in kf.split(X_with_E):
            # model with E
            hgbt_with_E = HistGradientBoostingClassifier(random_state=self.random_state)
            hgbt_with_E.fit(X_with_E[train_idx], y[train_idx])
            preds_with_E[test_idx] = hgbt_with_E.predict_proba(X_with_E[test_idx])[:, 1]

            # model without E (permuted E)
            hgbt_without_E = HistGradientBoostingClassifier(
                random_state=self.random_state
            )
            hgbt_without_E.fit(X_without_E[train_idx], y[train_idx])
            preds_without_E[test_idx] = hgbt_without_E.predict_proba(
                X_without_E[test_idx]
            )[:, 1]

        return preds_with_E, preds_without_E

    def test(self, X: np.ndarray, y: np.ndarray, E: np.ndarray) -> float:
        """
        Perform the DeLong test for invariance.

        Parameters
        ----------
        X : np.ndarray
            Predictor subset X_S of shape (n_samples, n_features).
            Can be empty (n_features=0) to test the empty set.
        y : np.ndarray
            Target variable of shape (n_samples,), assumed binary {0, 1}.
        E : np.ndarray
            Environment labels of shape (n_samples,).

        Returns
        -------
        p_value : float
            p-value of DeLong's test. Small values indicate that including E
            improves prediction, suggesting S is not invariant.
        """
        if len(np.unique(E)) < 2:
            # test undefined for single environment; assume invariance
            return 1.0

        # encode environment as numeric
        le = LabelEncoder()
        E_encoded = np.array(le.fit_transform(E)).reshape(-1, 1)

        # handle empty X case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] == 0:
            X_with_E = E_encoded
        else:
            X_with_E = np.hstack([X, E_encoded])

        # permute E for the "without E" model
        rng = np.random.RandomState(self.random_state)
        E_permuted = rng.permutation(E_encoded)

        if X.shape[1] == 0:
            X_without_E = E_permuted
        else:
            X_without_E = np.hstack([X, E_permuted])

        # get predictions based on classifier type
        try:
            if self.test_classifier_type == "RF":
                preds_with_E, preds_without_E = self._get_predictions_rf(
                    X_with_E, X_without_E, y
                )
            elif self.test_classifier_type == "HGBT":
                preds_with_E, preds_without_E = self._get_predictions_hgbt(
                    X_with_E, X_without_E, y
                )
            else:  # LR
                preds_with_E, preds_without_E = self._get_predictions_lr(
                    X_with_E, X_without_E, y
                )
        except Exception:
            # ff fitting fails, return 1.0 (cannot reject invariance)
            return 1.0

        # handle NaN predictions
        if np.any(np.isnan(preds_with_E)) or np.any(np.isnan(preds_without_E)):
            return 1.0

        # perform DeLong's test
        # H0: AUC_without_E = AUC_with_E
        # H1: AUC_without_E < AUC_with_E (E helps, so not invariant)
        p_value = delong_test(y, preds_without_E, preds_with_E, alternative="less")

        if np.isnan(p_value):
            return 1.0

        return p_value
