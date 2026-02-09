"""
Evaluate stabilized classification on causal chambers datasets.

Fits StabilizedClassificationClassifier on a *_train.csv dataset and evaluates
on the corresponding *_test.csv dataset.  Three invariance-test variants are
used (Residual LR, CRT HGBT, TramGCM RF), each with a Random Forest
prediction model.  Results are compared against a standard (pooled) Random
Forest and random guessing.

Metrics reported per test-environment and in summary:
  - Binary cross-entropy (BCE) loss   (lower is better)
  - Accuracy                          (higher is better)

================================================================================
USAGE
================================================================================

    python evaluate_stabclass.py --dataset 1_small
    python evaluate_stabclass.py --dataset 1

The script expects <name>_train.csv and <name>_test.csv in the data/ directory.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

from invariance_for_classification import StabilizedClassificationClassifier

# ── stable blanket definitions ────────────────────────────────────────────────

# The stable blanket is the theoretically optimal subset of predictors.
# It contains exactly those features whose relationship with Y is invariant
# across environments and that are maximally predictive.
STABLE_BLANKETS: dict[str, list[str]] = {
    "1_small": ["red", "green", "blue", "vis_3"],
    "1": ["red", "green", "blue", "vis_3"],
    "2_original_small": ["red", "green", "blue", "vis_3"],
    "2_original": ["red", "green", "blue", "vis_3"],
    "2_v2_small": ["red", "green", "blue", "vis_3"],
    "2_v2": ["red", "green", "blue", "vis_3"],
    "2_v3_small": ["red", "green", "blue", "vis_3"],
    "2_v3": ["red", "green", "blue", "vis_3"],
    "2_v4_small": ["red", "green", "blue", "vis_3"],
    "2_v4": ["red", "green", "blue", "vis_3"],
}


def _get_stable_blanket(dataset: str) -> list[str] | None:
    """Return the stable blanket for a dataset, or None if unknown."""
    return STABLE_BLANKETS.get(dataset)


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_train_test(
    dataset: str, data_dir: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_train, df_test) for a given dataset base name."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    train_path = os.path.join(data_dir, f"{dataset}_train.csv")
    test_path = os.path.join(data_dir, f"{dataset}_test.csv")

    if not os.path.exists(train_path):
        sys.exit(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        sys.exit(f"Test file not found: {test_path}")

    return pd.read_csv(train_path), pd.read_csv(test_path)


def _per_env_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    envs: np.ndarray,
) -> pd.DataFrame:
    """Compute BCE loss and accuracy for each environment."""
    records = []
    for e in np.sort(np.unique(envs)):
        mask = envs == e
        bce = float(log_loss(y_true[mask], np.clip(y_prob[mask], 0, 1), labels=[0, 1]))
        acc = float(accuracy_score(y_true[mask], y_pred[mask]))
        records.append(
            {"env": e, "bce_loss": bce, "accuracy": acc, "n": int(mask.sum())}
        )
    return pd.DataFrame(records)


def _summary_row(per_env: pd.DataFrame, model_name: str) -> dict:
    """Aggregate per-environment metrics into a summary row."""
    return {
        "model": model_name,
        "mean_bce": per_env["bce_loss"].mean(),
        "max_bce": per_env["bce_loss"].max(),
        "mean_acc": per_env["accuracy"].mean(),
        "min_acc": per_env["accuracy"].min(),
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main(dataset: str, n_jobs: int = 10) -> None:
    print("=" * 70)
    print("STABILIZED CLASSIFICATION - EVALUATION")
    print("=" * 70)

    # ── load data ────────────────────────────────────────────────────────
    df_train, df_test = _load_train_test(dataset)
    features = [c for c in df_train.columns if c not in ("Y", "E")]

    stable_blanket = _get_stable_blanket(dataset)

    print(f"\nDataset        : {dataset}")
    print(f"Features       : {features}")
    print(f"Stable blanket : {stable_blanket if stable_blanket else '(unknown)'}")
    print(
        f"Train samples  : {len(df_train)}  (environments: {sorted(df_train['E'].unique())})"
    )
    print(
        f"Test samples   : {len(df_test)}  (environments: {sorted(df_test['E'].unique())})"
    )

    X_train = df_train[features].to_numpy()
    y_train = df_train["Y"].to_numpy()
    E_train = df_train["E"].to_numpy()

    X_test = df_test[features].to_numpy()
    y_test = df_test["Y"].to_numpy()
    E_test = df_test["E"].to_numpy()

    # containers for summary
    summaries: list[dict] = []

    # ── 1. Stabilized classifiers ────────────────────────────────────────
    stabclass_configs = [
        {
            "name": "StabClass (Residual LR)",
            "invariance_test": "inv_residual",
            "test_classifier_type": "LR",
        },
        {
            "name": "StabClass (CRT HGBT)",
            "invariance_test": "crt",
            "test_classifier_type": "HGBT",
        },
        {
            "name": "StabClass (TramGCM RF)",
            "invariance_test": "tram_gcm",
            "test_classifier_type": "RF",
        },
    ]

    for cfg in stabclass_configs:
        print(f"\n{'─' * 70}")
        print(f"Fitting {cfg['name']} …")
        print(f"{'─' * 70}")

        clf = StabilizedClassificationClassifier(
            invariance_test=cfg["invariance_test"],
            test_classifier_type=cfg["test_classifier_type"],
            pred_classifier_type="RF",
            n_jobs=n_jobs,
            random_state=42,
        )
        clf.fit(X_train, y_train, environment=E_train)

        # report which subsets were selected
        for stat in clf.active_subsets_:
            feat_names = (
                [features[i] for i in stat["subset"]] if stat["subset"] else ["∅"]
            )
            print(
                f"  Active subset: {feat_names}  (p={stat['p_value']:.4f}, score={stat['score']:.4f})"
            )

        y_prob = clf.predict_proba(X_test)  # shape (n, 2)
        y_pred = clf.predict(X_test)

        per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
        _print_per_env(cfg["name"], per_env)
        summaries.append(_summary_row(per_env, cfg["name"]))

    # ── 2. Standard Random Forest (pooled, no E) ────────────────────────
    print(f"\n{'─' * 70}")
    print("Fitting Standard Random Forest (pooled) …")
    print(f"{'─' * 70}")

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs)
    rf.fit(X_train, y_train)

    y_prob_rf = rf.predict_proba(X_test)
    y_pred_rf = rf.predict(X_test)

    per_env_rf = _per_env_metrics(y_test, y_prob_rf, y_pred_rf, E_test)
    _print_per_env("Standard RF (all)", per_env_rf)
    summaries.append(_summary_row(per_env_rf, "Standard RF (all)"))

    # ── 3. Oracle RF on stable blanket ──────────────────────────────────
    stable_blanket = _get_stable_blanket(dataset)
    if stable_blanket is not None:
        print(f"\n{'─' * 70}")
        print(f"Fitting Oracle RF on stable blanket {stable_blanket} …")
        print(f"{'─' * 70}")

        sb_idx = [features.index(f) for f in stable_blanket]
        X_train_sb = X_train[:, sb_idx]
        X_test_sb = X_test[:, sb_idx]

        rf_sb = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs)
        rf_sb.fit(X_train_sb, y_train)

        y_prob_sb = rf_sb.predict_proba(X_test_sb)
        y_pred_sb = rf_sb.predict(X_test_sb)

        per_env_sb = _per_env_metrics(y_test, y_prob_sb, y_pred_sb, E_test)
        _print_per_env("Oracle RF (stable blanket)", per_env_sb)
        summaries.append(_summary_row(per_env_sb, "Oracle RF (stable blanket)"))
    else:
        print(
            f"\n  (No stable blanket defined for dataset '{dataset}' – skipping oracle RF)"
        )

    # ── 4. Random guessing baseline ─────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Random guessing baseline")
    print(f"{'─' * 70}")

    # Use the training class prior as the constant predicted probability
    prior = y_train.mean()
    n_test = len(y_test)
    y_prob_rand = np.column_stack([np.full(n_test, 1 - prior), np.full(n_test, prior)])
    # For predicted labels: assign the majority class
    y_pred_rand = np.full(n_test, int(prior >= 0.5))

    per_env_rand = _per_env_metrics(y_test, y_prob_rand, y_pred_rand, E_test)
    _print_per_env("Random Guessing", per_env_rand)
    summaries.append(_summary_row(per_env_rand, "Random Guessing"))

    # ── 5. Summary table ────────────────────────────────────────────────
    summary_df = pd.DataFrame(summaries)

    print(f"\n{'=' * 70}")
    print("SUMMARY  (BCE loss: lower is better | Accuracy: higher is better)")
    print("=" * 70)
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
            columns=["model", "mean_bce", "max_bce", "mean_acc", "min_acc"],
            header=["Model", "Mean BCE ↓", "Max BCE ↓", "Mean Acc ↑", "Min Acc ↑"],
        )
    )
    print()


def _print_per_env(model_name: str, per_env: pd.DataFrame) -> None:
    """Pretty-print per-environment results."""
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
        f"  {'Mean':>5s}  {'':>5s}  {per_env['bce_loss'].mean():12.4f}  {per_env['accuracy'].mean():12.4f}\n"
        f"  {'Max':>5s}   {'':>5s}  {per_env['bce_loss'].max():12.4f}  {per_env['accuracy'].min():12.4f}  (worst env)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate stabilized classification on causal chambers datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="1_small",
        help="Dataset base name (e.g. '1_small' expects 1_small_train.csv / 1_small_test.csv)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=10,
        help="Number of parallel jobs for fitting (default: 10)",
    )
    args = parser.parse_args()
    main(dataset=args.dataset, n_jobs=args.n_jobs)
