"""
LOEO Regret (Leave-One-Environment-Out Regret).

For each held-out environment *e* and feature subset *S*:

  1. Train environment-specific classifiers f_S^{e'} for every e' ≠ e.
  2. Form the LOEO average (ensemble) classifier

         f_S^{-e}(x) = (1 / |E|-1) Σ_{e'≠e} f_S^{e'}(x)

  3. Compute the LOEO regret in environment *e*:

         R_e^LOEO(S) = R_e(f_S^{-e}) - (1 / |E|-1) Σ_{e'≠e} R_e(f_S^{e'})

     where R_e(·) is the mean BCE loss evaluated on data from environment e.

Returns a dictionary with aggregate scores (mean and min of regrets).
Higher values (closer to 0) indicate "more invariant".

Intuition: If Y | X_S is truly invariant across environments, averaging the
predicted probabilities of environment-specific models should not hurt
compared with using them individually—the ensemble prediction converges to
the shared conditional. A large negative regret signals that individual
models disagree, i.e. the conditional is environment-specific.
"""

from typing import Literal, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _binary_cross_entropy(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15
) -> np.ndarray:
    """
    Compute element-wise binary cross-entropy loss.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted probabilities for class 1.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    np.ndarray
        Element-wise BCE loss values.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def _make_classifier(
    classifier_type: Literal["RF", "HGBT", "LR"] = "RF",
    random_state: Optional[int] = 42,
    n_estimators: int = 100,
) -> RandomForestClassifier | HistGradientBoostingClassifier | LogisticRegression:
    """
    Create a classifier with appropriate regularisation.

    Parameters
    ----------
    classifier_type : {"RF", "HGBT", "LR"}, default="RF"
        Classifier type to create.
    random_state : int or None, default=42
        Random seed for reproducibility.
    n_estimators : int, default=100
        Number of trees (RF only).

    Returns
    -------
    RandomForestClassifier, HistGradientBoostingClassifier, or LogisticRegression
    """
    if classifier_type == "HGBT":
        return HistGradientBoostingClassifier(
            random_state=random_state,
            min_samples_leaf=20,
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
        )
    if classifier_type == "LR":
        return LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            C=1e10,  # approximate unpenalized logistic regression
        )
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,
    )


def loeo_regret(
    Y: np.ndarray,
    E: np.ndarray,
    X_S: np.ndarray,
    n_estimators: int = 100,
    random_state: Optional[int] = 42,
    classifier_type: Literal["RF", "HGBT", "LR"] = "HGBT",
) -> dict[str, float]:
    """
    Compute LOEO Regret 2 (ensemble variant) scores.

    For each held-out environment *e*:

      1. For every other environment *e'*, train an environment-specific model
         on *e'* and obtain predicted probabilities on *e* → ``preds_{e'}``.
      2. Form the ensemble prediction by averaging:
         ``ensemble_preds_e = mean(preds_{e'}  for e' ≠ e)``.
      3. Compute BCE loss of ensemble predictions on *e* → ``ensemble_loss_e``.
      4. Compute BCE loss of each individual model on *e* → ``env_loss_{e'}``.
      5. ``regret_e = ensemble_loss_e - mean(env_loss_{e'}  for e' ≠ e)``.

    Returns a dictionary containing aggregated scores based on the regrets:
      - 'mean': Mean of regrets across held-out environments.
      - 'min': Minimum of regrets across held-out environments.

    Higher values (closer to 0) indicate more invariance. A very negative
    regret means the averaged (ensemble) prediction performed much worse than
    individual environment-specific models, signalling environment-specificity.

    Parameters
    ----------
    Y : np.ndarray of shape (n_samples,)
        Target variable, binary {0, 1}.
    E : np.ndarray of shape (n_samples,)
        Environment indicator.
    X_S : np.ndarray of shape (n_samples, n_features)
        Subset of predictors to evaluate.
    n_estimators : int, default=100
        Number of trees in the random forest (only used for RF).
    random_state : int or None, default=42
        Random seed for reproducibility.
    classifier_type : {"RF", "HGBT", "LR"}, default="HGBT"
        Classifier type to use. HGBT is recommended for best performance.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'mean' and 'min'.
        Returns {'mean': 0.0, 'min': 0.0} for edge cases (single environment).
    """
    unique_envs = np.unique(E)
    n_envs = len(unique_envs)

    # single environment case: cannot compute regrets, return 0.0
    if n_envs < 2:
        return {"mean": 0.0, "min": 0.0}

    Y = np.asarray(Y).astype(int)
    E = np.asarray(E)

    # --- fit one model per environment once and store it ---
    fitted_models: dict = {}
    for e_train in unique_envs:
        mask_e = E == e_train
        if X_S.shape[1] == 0:
            # no features: just store the marginal rate
            fitted_models[e_train] = float(np.mean(Y[mask_e]))
        else:
            clf = _make_classifier(classifier_type, random_state, n_estimators)
            y_e = Y[mask_e]
            if len(np.unique(y_e)) < 2:
                # single-class environment: store marginal rate
                fitted_models[e_train] = float(np.mean(y_e))
            else:
                clf.fit(X_S[mask_e], y_e)
                fitted_models[e_train] = clf

    # --- pre-compute all pairwise predictions: preds[e_train][e_test] ---
    # preds[e'][e] = predicted P(Y=1) by model trained on e', evaluated on e
    preds: dict[object, dict[object, np.ndarray]] = {}
    for e_train in unique_envs:
        preds[e_train] = {}
        model = fitted_models[e_train]
        for e_test in unique_envs:
            if e_test == e_train:
                continue
            mask_test = E == e_test
            if isinstance(model, float):
                preds[e_train][e_test] = np.full(int(mask_test.sum()), model)
            else:
                preds[e_train][e_test] = model.predict_proba(X_S[mask_test])[:, 1]

    regrets: list[float] = []

    for e in unique_envs:
        y_test = Y[E == e]
        other_envs = [e2 for e2 in unique_envs if e2 != e]

        # --- individual env losses and stacked predictions ---
        all_preds = np.stack([preds[e2][e] for e2 in other_envs], axis=0)
        env_losses = [
            float(np.mean(_binary_cross_entropy(y_test, preds[e2][e])))
            for e2 in other_envs
        ]

        # --- ensemble: average of environment-specific predictions ---
        ensemble_preds = np.mean(all_preds, axis=0)
        ensemble_loss = float(np.mean(_binary_cross_entropy(y_test, ensemble_preds)))

        avg_env_loss = float(np.mean(env_losses))
        regrets.append(ensemble_loss - avg_env_loss)

    regrets_arr = np.array(regrets)

    # check for NaN values (can happen with edge cases)
    if np.any(np.isnan(regrets_arr)):
        return {"mean": 0.0, "min": 0.0}

    return {
        "mean": float(np.mean(regrets_arr)),
        "min": float(np.min(regrets_arr)),
    }
