"""
Evaluation of multiple methods on causal chambers datasets.

Compares:
  1. Stabilized Classification (our method)
  2. IRM (Invariant Risk Minimization)
  3. V-REx (Variance Risk Extrapolation)
  4. ERM Neural Network
  5. Random Forest (pooled)
  6. Random Forest on Stable Blanket (oracle)
  7. ICP-glm (Invariant Causal Prediction, logistic regression)
  8. ICP-rf (Invariant Causal Prediction, random forest)

usage:
    python evaluate_all.py --dataset 1_v2 --n-jobs 10
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import (
    ERMClassifier,
    ICPglmClassifier,
    ICPrfClassifier,
    IRMClassifier,
    VRExClassifier,
)

from invariance_for_classification import StabilizedClassificationClassifier

# ──────────────────────────────────────────────────────────────────────────────
# stable blanket definitions (ground truth for oracle comparison)
# ──────────────────────────────────────────────────────────────────────────────

STABLE_BLANKETS: dict[str, list[str]] = {
    "1_v1_small": ["red", "green", "blue", "vis_3"],
    "1_v1": ["red", "green", "blue", "vis_3"],
    "1_v2_small": ["red", "green", "blue", "vis_3"],
    "1_v2": ["red", "green", "blue", "vis_3"],
    "1_v3_small": ["red", "green", "blue", "vis_3"],
    "1_v3": ["red", "green", "blue", "vis_3"],
    "5_v1_small": ["red", "green", "blue"],
    "5_v1": ["red", "green", "blue"],
    "5_v2_small": ["red", "green", "blue"],
    "5_v2": ["red", "green", "blue"],
}


# ──────────────────────────────────────────────────────────────────────────────
# data loading
# ──────────────────────────────────────────────────────────────────────────────


def _load_train_test(
    dataset: str, data_dir: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs for a dataset."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_path = os.path.join(data_dir, f"{dataset}_train.csv")
    test_path = os.path.join(data_dir, f"{dataset}_test.csv")
    if not os.path.exists(train_path):
        sys.exit(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        sys.exit(f"Test file not found: {test_path}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


# ──────────────────────────────────────────────────────────────────────────────
# eval helpers
# ──────────────────────────────────────────────────────────────────────────────


def _extract_positive_proba(y_proba: np.ndarray) -> np.ndarray:
    if y_proba.ndim == 2:
        return y_proba[:, 1]
    return y_proba


def _per_env_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    envs: np.ndarray,
) -> pd.DataFrame:
    """Compute BCE loss and accuracy for each environment.

    y_prob should be the prob of the positive class.
    """
    records = []
    for e in np.sort(np.unique(envs)):
        mask = envs == e
        bce = float(
            log_loss(y_true[mask], np.clip(y_prob[mask], 1e-7, 1 - 1e-7), labels=[0, 1])
        )
        acc = float(accuracy_score(y_true[mask], y_pred[mask]))
        records.append(
            {"env": e, "bce_loss": bce, "accuracy": acc, "n": int(mask.sum())}
        )
    return pd.DataFrame(records)


def _summary_row(per_env: pd.DataFrame, model_name: str) -> dict:
    """Create a summary row from per-environment metrics."""
    return {
        "model": model_name,
        "mean_bce": per_env["bce_loss"].mean(),
        "max_bce": per_env["bce_loss"].max(),
        "mean_acc": per_env["accuracy"].mean(),
        "min_acc": per_env["accuracy"].min(),
    }


def _print_per_env(model_name: str, per_env: pd.DataFrame) -> None:
    """Print per-environment results."""
    print(f"\n  Per-environment results for {model_name}:")
    print(f"  {'Env':>5s}  {'n':>5s}  {'BCE loss ↓':>12s}  {'Accuracy ↑':>12s}")
    print(f"  {'─' * 40}")
    for _, row in per_env.iterrows():
        print(
            f"  {int(row['env']):5d}  {int(row['n']):5d}  "
            f"{row['bce_loss']:12.4f}  {row['accuracy']:12.4f}"
        )
    print(
        f"  {'─' * 40}\n"
        f"  {'Mean':>5s}  {'':>5s}  {per_env['bce_loss'].mean():12.4f}  "
        f"{per_env['accuracy'].mean():12.4f}\n"
        f"  {'Worst':>5s}  {'':>5s}  {per_env['bce_loss'].max():12.4f}  "
        f"{per_env['accuracy'].min():12.4f}  (worst env)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# method defs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MethodResult:
    """Result from fitting and evaluating a method."""

    name: str
    per_env: pd.DataFrame
    summary: dict


def evaluate_stabilized_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    n_jobs: int = 1,
    verbose: bool = False,
    pred_scoring: str = "pooled",
) -> MethodResult:
    """Fit and evaluate Stabilized Classification."""
    name = f"StabClass ({pred_scoring})"

    clf = StabilizedClassificationClassifier(
        alpha_inv=0.05,
        alpha_pred=0.05,
        pred_classifier_type="RF",
        test_classifier_type="RF",
        invariance_test="tram_gcm",
        pred_scoring=pred_scoring,
        n_bootstrap=250,
        verbose=1 if verbose else 0,
        n_jobs=n_jobs,
    )

    clf.fit(X_train, y_train, environment=E_train)

    y_prob = _extract_positive_proba(clf.predict_proba(X_test))
    y_pred = clf.predict(X_test)

    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    if verbose:
        _print_per_env(name, per_env)

    return MethodResult(
        name=name,
        per_env=per_env,
        summary=_summary_row(per_env, name),
    )


def evaluate_irm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    n_jobs: int = 1,
    verbose: bool = False,
) -> MethodResult:
    """Fit and evaluate IRM."""
    name = "IRM"
    clf = IRMClassifier(n_jobs=n_jobs, verbose=verbose)

    clf.fit(X_train, y_train, environment=E_train)

    y_prob = _extract_positive_proba(clf.predict_proba(X_test))
    y_pred = clf.predict(X_test)

    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    if verbose:
        _print_per_env(name, per_env)

    return MethodResult(
        name=name,
        per_env=per_env,
        summary=_summary_row(per_env, name),
    )


def evaluate_vrex(
    X_train: np.ndarray,
    y_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    n_jobs: int = 1,
    verbose: bool = False,
) -> MethodResult:
    """Fit and evaluate V-REx."""
    name = "V-REx"
    clf = VRExClassifier(n_jobs=n_jobs, verbose=verbose)

    clf.fit(X_train, y_train, environment=E_train)

    y_prob = _extract_positive_proba(clf.predict_proba(X_test))
    y_pred = clf.predict(X_test)

    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    if verbose:
        _print_per_env(name, per_env)

    return MethodResult(
        name=name,
        per_env=per_env,
        summary=_summary_row(per_env, name),
    )


def evaluate_erm_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    n_jobs: int = 1,
    verbose: bool = False,
) -> MethodResult:
    """Fit and evaluate ERM Neural Network."""
    name = "ERM (NN)"
    clf = ERMClassifier(n_jobs=n_jobs, verbose=verbose)

    clf.fit(X_train, y_train, environment=E_train)

    y_prob = _extract_positive_proba(clf.predict_proba(X_test))
    y_pred = clf.predict(X_test)

    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    if verbose:
        _print_per_env(name, per_env)

    return MethodResult(
        name=name,
        per_env=per_env,
        summary=_summary_row(per_env, name),
    )


def evaluate_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    n_jobs: int = 1,
    verbose: bool = False,
) -> MethodResult:
    """Fit and evaluate pooled Random Forest (ignores environments)."""
    name = "RF (pooled)"
    clf = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=42)

    clf.fit(X_train, y_train)

    y_prob = _extract_positive_proba(clf.predict_proba(X_test))
    y_pred = clf.predict(X_test)

    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    if verbose:
        _print_per_env(name, per_env)

    return MethodResult(
        name=name,
        per_env=per_env,
        summary=_summary_row(per_env, name),
    )


def evaluate_random_forest_oracle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    stable_blanket_indices: list[int],
    n_jobs: int = 1,
    verbose: bool = False,
) -> MethodResult:
    """Fit and evaluate Random Forest on stable blanket (oracle)."""
    name = "RF (oracle)"

    X_train_sb = X_train[:, stable_blanket_indices]
    X_test_sb = X_test[:, stable_blanket_indices]

    clf = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=42)

    clf.fit(X_train_sb, y_train)

    y_prob = _extract_positive_proba(clf.predict_proba(X_test_sb))
    y_pred = clf.predict(X_test_sb)

    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    if verbose:
        _print_per_env(name, per_env)

    return MethodResult(
        name=name,
        per_env=per_env,
        summary=_summary_row(per_env, name),
    )


def evaluate_icp_glm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    n_jobs: int = 1,
    verbose: bool = False,
) -> MethodResult:
    """Fit and evaluate ICP with glmICP (logistic regression)."""
    name = "ICP (glm)"

    clf = ICPglmClassifier(alpha=0.05, n_jobs=n_jobs, verbose=verbose)
    clf.fit(X_train, y_train, environment=E_train)

    y_prob = _extract_positive_proba(clf.predict_proba(X_test))
    y_pred = clf.predict(X_test)

    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    if verbose:
        _print_per_env(name, per_env)
        print(f"  Invariant set indices: {clf.invariant_indices_}")

    return MethodResult(
        name=name,
        per_env=per_env,
        summary=_summary_row(per_env, name),
    )


def evaluate_icp_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    n_jobs: int = 1,
    verbose: bool = False,
) -> MethodResult:
    """Fit and evaluate ICP with rangerICP (random forest)."""
    name = "ICP (rf)"

    clf = ICPrfClassifier(alpha=0.05, n_jobs=n_jobs, verbose=verbose)
    clf.fit(X_train, y_train, environment=E_train)

    y_prob = _extract_positive_proba(clf.predict_proba(X_test))
    y_pred = clf.predict(X_test)

    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    if verbose:
        _print_per_env(name, per_env)
        print(f"  Invariant set indices: {clf.invariant_indices_}")

    return MethodResult(
        name=name,
        per_env=per_env,
        summary=_summary_row(per_env, name),
    )


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────


def main(dataset: str, n_jobs: int = 1, verbose: bool = True) -> pd.DataFrame:
    """Run evaluation on a dataset and return summary DataFrame."""
    print("=" * 70)
    print("Evaluation of multiple methods")
    print("=" * 70)

    # load data
    df_train, df_test = _load_train_test(dataset)
    features = [c for c in df_train.columns if c not in ("Y", "E")]

    print(f"\nDataset        : {dataset}")
    print(f"Features ({len(features):d})   : {features}")
    print(f"Train samples  : {len(df_train)}  (envs: {sorted(df_train['E'].unique())})")
    print(f"Test samples   : {len(df_test)}  (envs: {sorted(df_test['E'].unique())})")

    # extract arrays
    X_train = df_train[features].to_numpy()
    y_train = df_train["Y"].to_numpy().astype(int)
    E_train = df_train["E"].to_numpy()

    X_test = df_test[features].to_numpy()
    y_test = df_test["Y"].to_numpy().astype(int)
    E_test = df_test["E"].to_numpy()

    # get stable blanket if available
    stable_blanket = STABLE_BLANKETS.get(dataset)
    stable_blanket_indices: list[int] | None = None
    if stable_blanket is not None:
        stable_blanket_indices = [
            features.index(f) for f in stable_blanket if f in features
        ]
        print(f"Stable blanket : {stable_blanket}")
    else:
        print("Stable blanket : (unknown)")

    # evaluate all methods
    results: list[MethodResult] = []

    # 1. Stabilized Classification (our method) - all scoring strategies
    for pred_scoring in ["pooled", "worst_case"]:
        try:
            results.append(
                evaluate_stabilized_classification(
                    X_train,
                    y_train,
                    E_train,
                    X_test,
                    y_test,
                    E_test,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    pred_scoring=pred_scoring,
                )
            )
        except Exception as e:
            print(f"  StabClass ({pred_scoring}) failed: {e}")

    # 2. IRM
    try:
        results.append(
            evaluate_irm(
                X_train,
                y_train,
                E_train,
                X_test,
                y_test,
                E_test,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        )
    except Exception as e:
        print(f"  IRM failed: {e}")

    # 3. V-REx
    try:
        results.append(
            evaluate_vrex(
                X_train,
                y_train,
                E_train,
                X_test,
                y_test,
                E_test,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        )
    except Exception as e:
        print(f"  V-REx failed: {e}")

    # 4. ERM (NN)
    try:
        results.append(
            evaluate_erm_nn(
                X_train,
                y_train,
                E_train,
                X_test,
                y_test,
                E_test,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        )
    except Exception as e:
        print(f"  ERM (NN) failed: {e}")

    # 5. Random Forest (pooled)
    try:
        results.append(
            evaluate_random_forest(
                X_train,
                y_train,
                X_test,
                y_test,
                E_test,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        )
    except Exception as e:
        print(f"  RF (pooled) failed: {e}")

    # 6. Random Forest on stable blanket (oracle)
    if stable_blanket_indices is not None and len(stable_blanket_indices) > 0:
        try:
            results.append(
                evaluate_random_forest_oracle(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    E_test,
                    stable_blanket_indices=stable_blanket_indices,
                    n_jobs=n_jobs,
                    verbose=verbose,
                )
            )
        except Exception as e:
            print(f"  RF (oracle) failed: {e}")

    # 7. ICP (glm) - logistic regression-based
    try:
        results.append(
            evaluate_icp_glm(
                X_train,
                y_train,
                E_train,
                X_test,
                y_test,
                E_test,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        )
    except Exception as e:
        print(f"  ICP (glm) failed: {e}")

    # 8. ICP (rf) - random forest-based
    try:
        results.append(
            evaluate_icp_rf(
                X_train,
                y_train,
                E_train,
                X_test,
                y_test,
                E_test,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        )
    except Exception as e:
        print(f"  ICP (rf) failed: {e}")

    summaries = [r.summary for r in results]
    summary_df = pd.DataFrame(summaries)

    print(f"\n{'=' * 70}")
    print("SUMMARY  (BCE loss: lower is better | Accuracy: higher is better)")
    print("=" * 70)
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
            columns=[
                "model",
                "mean_bce",
                "max_bce",
                "mean_acc",
                "min_acc",
            ],
            header=[
                "Model",
                "Mean BCE ↓",
                "Max BCE ↓",
                "Mean Acc ↑",
                "Min Acc ↑",
            ],
        )
    )
    print()

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluation of multiple methods on causal chambers datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="1_v2",
        help="Dataset base name, e.g. '1_v2'",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-method verbose output.",
    )
    args = parser.parse_args()
    main(dataset=args.dataset, n_jobs=args.n_jobs, verbose=not args.quiet)
