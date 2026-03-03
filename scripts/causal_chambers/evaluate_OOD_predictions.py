"""
Performance evaluation for stabilized classification and baseline methods on causal chambers
datasets.

For each of N_REPS repetitions:
  1. Draw a random subsample of N_OBS_PER_ENV observations per training environment from a
     causal chambers dataset
  2. Fit all methods (SC configurations + baselines) on the subsample
  3. Evaluate on the full test dataset

Metrics: accuracy and log-loss (BCE) per test environment, plus min accuracy
and max log-loss over environments.

Results are saved incrementally.

================================================================================
SC CONFIGURATIONS (invariance test, test classifier type)
================================================================================
  - CRT (RF)
  - DeLong (RF), DeLong (LR)
  - InvEnvPred (RF)
  - Residual (RF), Residual (LR)
  - TramGCM (RF), TramGCM (LR)
  - WGCM_est (xgb), WGCM_fix (xgb)
  - LOEO Regret (RF), LOEO Regret (LR)

Each SC configuration produces predictions with both RF and LR base classifiers.

================================================================================
BASELINES
================================================================================
  - ERM LR (pooled logistic regression)
  - ERM RF (pooled random forest)
  - Oracle LR (logistic regression on stable blanket)
  - Oracle RF (random forest on stable blanket)
  - IRM-NN (neural network IRM)
  - IRM-Linear (linear IRM from InvarianceUnitTests)

================================================================================
OUTPUT
================================================================================
Results are saved to: <repo_root>/results/<dataset>/

  - OOD_raw_<dataset>.csv          Per-rep, per-environment results (incremental)
  - OOD_sc_subsets_<dataset>.csv   Subset counts for SC configs (incremental)
  - OOD_summary_<dataset>.csv      Means and 95% t-CIs over repetitions

================================================================================
USAGE
================================================================================
    python evaluate_OOD_predictions.py --dataset 1a
    python evaluate_OOD_predictions.py --dataset 2 --n-obs 200 --n-reps 20 --n-jobs 10

    # All flags:
    python evaluate_OOD_predictions.py --dataset 1a --n-obs 200 --n-reps 20 --n-jobs 10 --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Limit threads for OpenMP-based libraries (XGBoost, sklearn) to allow proper
# parallelization at the process level. Must be set before importing them.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import IRMLinearClassifier, IRMNNClassifier

from invariance_for_classification import StabilizedClassificationClassifier

# =============================================================================
# global configuration
# =============================================================================

SEED = 42

STABLE_BLANKETS: dict[str, list[str]] = {
    "1a": ["red", "green", "blue", "vis_3"],
    "1b": ["red", "green", "blue", "vis_3"],
    "2": ["red", "green", "blue"],
}

NORMAL_COLS: dict[str, list[str]] = {
    "1a": ["Y", "red", "green", "blue", "ir_1", "vis_1", "ir_3", "vis_3", "E"],
    "1b": ["Y", "red", "green", "blue", "ir_1", "vis_1", "ir_3", "vis_3", "E"],
    "2": ["Y", "red", "green", "blue", "ir_2", "vis_2", "ir_3", "vis_3", "E"],
}

# SC invariance test configurations:
# (display_name, invariance_test, test_classifier_type or None)
# Limited to those appearing in the paper tables (CRT omitted).
SC_CONFIGS: list[tuple[str, str, str | None]] = [
    ("DeLong(RF)", "delong", "RF"),
    ("DeLong(LR)", "delong", "LR"),
    ("InvEnvPred(RF)", "inv_env_pred", "RF"),
    ("Residual(RF)", "inv_residual", "RF"),
    ("Residual(LR)", "inv_residual", "LR"),
    ("TramGCM(RF)", "tram_gcm", "RF"),
    ("TramGCM(LR)", "tram_gcm", "LR"),
    ("WGCM_est", "wgcm_est", None),
    ("WGCM_fix", "wgcm_fix", None),
    ("LOEO(RF)", "loeo_regret", "RF"),
    ("LOEO(LR)", "loeo_regret", "LR"),
]


# =============================================================================
# data loading
# =============================================================================


def _load_train_test(
    dataset: str,
    data_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs, filtered to NORMAL_COLS."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_path = os.path.join(data_dir, f"{dataset}_train.csv")
    test_path = os.path.join(data_dir, f"{dataset}_test.csv")
    if not os.path.exists(train_path):
        sys.exit(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        sys.exit(f"Test file not found: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    if dataset in NORMAL_COLS:
        cols = [c for c in NORMAL_COLS[dataset] if c in df_train.columns]
        df_train = df_train[cols]
        df_test = df_test[cols]

    return df_train, df_test


def _subsample_train(
    df_train: pd.DataFrame, n_obs_per_env: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Subsample exactly n_obs_per_env observations per environment."""
    dfs = []
    for env in sorted(df_train["E"].unique()):
        env_df = df_train[df_train["E"] == env]
        if len(env_df) <= n_obs_per_env:
            dfs.append(env_df)
        else:
            idx = rng.choice(len(env_df), size=n_obs_per_env, replace=False)
            dfs.append(env_df.iloc[idx])
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# evaluation helpers
# =============================================================================


def _extract_positive_proba(y_proba: np.ndarray) -> np.ndarray:
    """Extract P(Y=1) from predict_proba output."""
    if y_proba.ndim == 2:
        return np.clip(y_proba[:, 1], 0.0, 1.0)
    return np.clip(y_proba, 0.0, 1.0)


def _per_env_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    envs: np.ndarray,
) -> list[dict]:
    """Compute BCE loss and accuracy per environment."""
    records = []
    for e in np.sort(np.unique(envs)):
        mask = envs == e
        bce = float(
            log_loss(
                y_true[mask],
                y_prob[mask],
                labels=[0, 1],
            )
        )
        acc = float(accuracy_score(y_true[mask], y_pred[mask]))
        records.append(
            {"env": int(e), "bce_loss": bce, "accuracy": acc, "n": int(mask.sum())}
        )
    return records


def _rows_from_per_env(per_env: list[dict], method: str, rep: int) -> list[dict]:
    """Build per-env result rows from per-env metrics."""
    rows = []
    for r in per_env:
        rows.append(
            {
                "rep": rep,
                "method": method,
                "env": r["env"],
                "accuracy": r["accuracy"],
                "bce_loss": r["bce_loss"],
                "n": r["n"],
            }
        )
    return rows


def _append_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """Append DataFrame to CSV, creating with header if new."""
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode="a", header=header, index=False)


# =============================================================================
# SC configuration runner
# =============================================================================


def _run_sc_config(
    config_name: str,
    invariance_test: str,
    test_classifier_type: str | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    n_jobs: int,
    rep: int,
    rep_seed: int,
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Fit one SC configuration and evaluate with both RF and LR base classifiers.

    Returns
    -------
    result_rows : list[dict]
        Per-environment results for each (config, pred_clf) combination.
    subset_rows : list[dict]
        Subset count info for this config.
    """
    kwargs: dict = {
        "alpha_inv": 0.05,
        "alpha_pred": 0.05,
        "pred_classifier_type": ["RF", "LR"],
        "invariance_test": invariance_test,
        "pred_scoring": "mean",
        "n_bootstrap": 250,
        "verbose": 1 if verbose else 0,
        "n_jobs": n_jobs,
        "random_state": rep_seed,
    }
    if test_classifier_type is not None:
        kwargs["test_classifier_type"] = test_classifier_type

    clf = StabilizedClassificationClassifier(**kwargs)
    clf.fit(X_train, y_train, environment=E_train)

    # extract subset counts
    n_inv = clf.n_invariant_subsets_
    n_pred = clf.n_predictive_subsets_  # dict[str, int] when list was passed

    subset_rows = [
        {
            "rep": rep,
            "sc_config": config_name,
            "n_invariant": n_inv,
            "n_predictive_RF": n_pred["RF"] if isinstance(n_pred, dict) else n_pred,
            "n_predictive_LR": n_pred["LR"] if isinstance(n_pred, dict) else n_pred,
        }
    ]

    result_rows: list[dict] = []
    for pred_clf in ["RF", "LR"]:
        for method_label, method_kwarg in [("ensemble", "ensemble"), ("best", "best")]:
            if method_label == "ensemble":
                method_name = f"SC {config_name}, pred={pred_clf}"
            else:
                method_name = f"SC {config_name}, pred={pred_clf}, best"
            y_prob = _extract_positive_proba(
                clf.predict_proba(
                    X_test, pred_classifier_type=pred_clf, method=method_kwarg
                )
            )
            y_pred = clf.predict(
                X_test, pred_classifier_type=pred_clf, method=method_kwarg
            )
            per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
            result_rows.extend(_rows_from_per_env(per_env, method_name, rep))

    return result_rows, subset_rows


# =============================================================================
# baseline runners
# =============================================================================


def _run_erm_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    rep: int,
    rep_seed: int,
    n_jobs: int = 1,
) -> list[dict]:
    """Fit and evaluate pooled Random Forest (ERM)."""
    clf = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=rep_seed)
    clf.fit(X_train, y_train)
    y_prob = _extract_positive_proba(clf.predict_proba(X_test))
    y_pred = np.asarray(clf.predict(X_test))
    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    return _rows_from_per_env(per_env, "ERM RF", rep)


def _run_erm_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    rep: int,
    rep_seed: int,
) -> list[dict]:
    """Fit and evaluate pooled Logistic Regression (ERM)."""
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=rep_seed)),
        ]
    )
    clf.fit(X_train, y_train)
    y_prob = _extract_positive_proba(clf.predict_proba(X_test))
    y_pred = np.asarray(clf.predict(X_test))
    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    return _rows_from_per_env(per_env, "ERM LR", rep)


def _run_oracle_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    stable_indices: list[int],
    rep: int,
    rep_seed: int,
    n_jobs: int = 1,
) -> list[dict]:
    """Fit and evaluate RF on the stable blanket (oracle)."""
    clf = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=rep_seed)
    clf.fit(X_train[:, stable_indices], y_train)
    y_prob = _extract_positive_proba(clf.predict_proba(X_test[:, stable_indices]))
    y_pred = np.asarray(clf.predict(X_test[:, stable_indices]))
    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    return _rows_from_per_env(per_env, "Oracle RF", rep)


def _run_oracle_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    stable_indices: list[int],
    rep: int,
    rep_seed: int,
) -> list[dict]:
    """Fit and evaluate LR on the stable blanket (oracle)."""
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=rep_seed)),
        ]
    )
    clf.fit(X_train[:, stable_indices], y_train)
    y_prob = _extract_positive_proba(clf.predict_proba(X_test[:, stable_indices]))
    y_pred = np.asarray(clf.predict(X_test[:, stable_indices]))
    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    return _rows_from_per_env(per_env, "Oracle LR", rep)


def _run_irm_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    rep: int,
    rep_seed: int,
    n_jobs: int = 1,
    verbose: bool = False,
) -> list[dict]:
    """Fit and evaluate IRM (neural network)."""
    clf = IRMNNClassifier(n_jobs=n_jobs, verbose=verbose, random_state=rep_seed)
    clf.fit(X_train, y_train, environment=E_train)
    y_prob = _extract_positive_proba(clf.predict_proba(X_test))
    y_pred = np.asarray(clf.predict(X_test))
    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    return _rows_from_per_env(per_env, "IRM-NN", rep)


def _run_irm_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    E_test: np.ndarray,
    rep: int,
    rep_seed: int,
    n_jobs: int = 1,
    verbose: bool = False,
) -> list[dict]:
    """Fit and evaluate IRM-Linear (InvarianceUnitTests-style)."""
    clf = IRMLinearClassifier(n_jobs=n_jobs, verbose=verbose, random_state=rep_seed)
    clf.fit(X_train, y_train, environment=E_train)
    y_prob = _extract_positive_proba(clf.predict_proba(X_test))
    y_pred = np.asarray(clf.predict(X_test))
    per_env = _per_env_metrics(y_test, y_prob, y_pred, E_test)
    return _rows_from_per_env(per_env, "IRM-Linear", rep)


# =============================================================================
# summary computation
# =============================================================================


def _ci_half_width(values: np.ndarray, confidence: float = 0.95) -> float:
    """Compute the half-width of a t-test confidence interval."""
    n = len(values)
    if n < 2:
        return float("nan")
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(t_crit * se)


def compute_summary(
    raw_path: str,
    sc_subsets_path: str,
    summary_path: str,
) -> pd.DataFrame:
    """
    Read the raw per-rep CSV and compute means + 95% t-CIs over repetitions.

    Summary statistics (min accuracy, max BCE, mean accuracy, mean BCE) are
    computed from the per-environment rows rather than stored in the raw CSV.

    Also merges mean SC subset counts from the sc_subsets CSV.
    """
    df = pd.read_csv(raw_path)

    records = []

    for method in sorted(df["method"].unique()):
        row: dict = {"method": method}
        m_env = df[df["method"] == method]

        # --- min accuracy over environments (one value per rep) ---
        min_accs = m_env.groupby("rep")["accuracy"].min().to_numpy()
        row["min_acc_mean"] = (
            float(np.mean(min_accs)) if len(min_accs) > 0 else float("nan")
        )
        row["min_acc_ci"] = _ci_half_width(min_accs)

        # --- max BCE over environments (one value per rep) ---
        max_bces = m_env.groupby("rep")["bce_loss"].max().to_numpy()
        row["max_bce_mean"] = (
            float(np.mean(max_bces)) if len(max_bces) > 0 else float("nan")
        )
        row["max_bce_ci"] = _ci_half_width(max_bces)

        # --- mean accuracy / BCE over environments (one per rep) ---
        mean_accs = m_env.groupby("rep")["accuracy"].mean().to_numpy()
        mean_bces = m_env.groupby("rep")["bce_loss"].mean().to_numpy()
        row["mean_acc_mean"] = (
            float(np.mean(mean_accs)) if len(mean_accs) > 0 else float("nan")
        )
        row["mean_acc_ci"] = _ci_half_width(mean_accs)
        row["mean_bce_mean"] = (
            float(np.mean(mean_bces)) if len(mean_bces) > 0 else float("nan")
        )
        row["mean_bce_ci"] = _ci_half_width(mean_bces)

        # --- per-environment metrics ---
        envs = sorted(m_env["env"].unique())
        for env in envs:
            env_data = m_env[m_env["env"] == env]
            acc_vals = env_data["accuracy"].to_numpy()
            bce_vals = env_data["bce_loss"].to_numpy()
            row[f"acc_env{env}_mean"] = (
                float(np.mean(acc_vals)) if len(acc_vals) > 0 else float("nan")
            )
            row[f"acc_env{env}_ci"] = _ci_half_width(acc_vals)
            row[f"bce_env{env}_mean"] = (
                float(np.mean(bce_vals)) if len(bce_vals) > 0 else float("nan")
            )
            row[f"bce_env{env}_ci"] = _ci_half_width(bce_vals)

        records.append(row)

    summary_df = pd.DataFrame(records)

    # merge SC subset counts if available
    if os.path.exists(sc_subsets_path):
        sc_df = pd.read_csv(sc_subsets_path)
        if len(sc_df) > 0:
            subset_means = (
                sc_df.groupby("sc_config")
                .agg(
                    {
                        "n_invariant": "mean",
                        "n_predictive_RF": "mean",
                        "n_predictive_LR": "mean",
                    }
                )
                .reset_index()
                .rename(
                    columns={
                        "n_invariant": "mean_n_invariant",
                        "n_predictive_RF": "mean_n_predictive_RF",
                        "n_predictive_LR": "mean_n_predictive_LR",
                    }
                )
            )
            # SC methods follow the pattern "SC <config_name>, pred=<clf>"
            # map each config to its two method names
            sc_merge_rows = []
            for _, srow in subset_means.iterrows():
                config = srow["sc_config"]
                for pred_clf in ["RF", "LR"]:
                    sc_merge_rows.append(
                        {
                            "method": f"SC {config}, pred={pred_clf}",
                            "mean_n_invariant": srow["mean_n_invariant"],
                            "mean_n_predictive_RF": srow["mean_n_predictive_RF"],
                            "mean_n_predictive_LR": srow["mean_n_predictive_LR"],
                        }
                    )
            sc_merge_df = pd.DataFrame(sc_merge_rows)
            summary_df = summary_df.merge(sc_merge_df, on="method", how="left")

    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    return summary_df


# =============================================================================
# main
# =============================================================================


def main(
    dataset: str = "1a",
    n_obs_per_env: int = 200,
    n_reps: int = 20,
    n_jobs: int = 10,
    verbose: bool = False,
) -> None:
    """Run the full OOD performance evaluation."""

    # ── reproducibility ──────────────────────────────────────────────────
    # No global np.random.seed(): every method receives its own rep_seed
    # and subsampling uses np.random.default_rng(rep_seed).
    # Only torch needs a global seed for deterministic CUDA behaviour.
    try:
        import torch

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except ImportError:
        pass

    # ── output directory ─────────────────────────────────────────────────
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    save_dir = os.path.join(repo_root, "results", dataset)
    os.makedirs(save_dir, exist_ok=True)

    raw_path = os.path.join(save_dir, f"OOD_raw_{dataset}.csv")
    sc_subsets_path = os.path.join(save_dir, f"OOD_sc_subsets_{dataset}.csv")
    summary_path = os.path.join(save_dir, f"OOD_summary_{dataset}.csv")

    # Remove stale output files so every run starts fresh
    for p in (raw_path, sc_subsets_path, summary_path):
        if os.path.exists(p):
            os.remove(p)

    # ── load data ────────────────────────────────────────────────────────
    df_train_full, df_test = _load_train_test(dataset)
    features = [c for c in df_train_full.columns if c not in ("Y", "E")]

    print("=" * 70)
    print("OOD Performance Evaluation")
    print("=" * 70)
    print(f"Dataset          : {dataset}")
    print(f"Features ({len(features):d})     : {features}")
    print(f"Full train size  : {len(df_train_full)}")
    print(f"Test size        : {len(df_test)}")
    print(f"N_OBS_PER_ENV    : {n_obs_per_env}")
    print(f"N_REPS           : {n_reps}")
    print(f"N_JOBS           : {n_jobs}")
    print(f"Save directory   : {save_dir}")

    # test arrays (constant across reps)
    X_test = df_test[features].to_numpy()
    y_test = df_test["Y"].to_numpy().astype(int)
    E_test = df_test["E"].to_numpy()

    # stable blanket indices for oracle baselines
    stable_blanket = STABLE_BLANKETS.get(dataset)
    stable_indices: list[int] | None = None
    if stable_blanket is not None:
        stable_indices = [features.index(f) for f in stable_blanket if f in features]
        print(f"Stable blanket   : {stable_blanket} -> indices {stable_indices}")
    else:
        print("Stable blanket   : (unknown)")

    # ── main loop ────────────────────────────────────────────────────────
    t_start_total = time.time()

    for rep in range(n_reps):
        rep_seed = SEED + rep
        rng = np.random.default_rng(rep_seed)

        print(f"\n{'─' * 70}")
        print(f"Repetition {rep + 1}/{n_reps}  (seed={rep_seed})")
        print(f"{'─' * 70}")

        # subsample training data
        df_train = _subsample_train(df_train_full, n_obs_per_env, rng)
        X_train = df_train[features].to_numpy()
        y_train = df_train["Y"].to_numpy().astype(int)
        E_train = df_train["E"].to_numpy()

        print(
            f"  Train subsample: {len(df_train)} obs "
            f"(envs: {sorted(df_train['E'].unique())})"
        )

        # ── SC configurations ─────────────────────────────────────────
        for cfg_name, inv_test, test_clf in SC_CONFIGS:
            t0 = time.time()
            print(f"  Running SC {cfg_name} ...", end=" ", flush=True)
            result_rows, subset_rows = _run_sc_config(
                config_name=cfg_name,
                invariance_test=inv_test,
                test_classifier_type=test_clf,
                X_train=X_train,
                y_train=y_train,
                E_train=E_train,
                X_test=X_test,
                y_test=y_test,
                E_test=E_test,
                n_jobs=n_jobs,
                rep=rep,
                rep_seed=rep_seed,
                verbose=verbose,
            )
            _append_to_csv(pd.DataFrame(result_rows), raw_path)
            _append_to_csv(pd.DataFrame(subset_rows), sc_subsets_path)
            print(f"done ({time.time() - t0:.1f}s)")

        # ── ERM baselines ─────────────────────────────────────────────
        _baseline_tasks = [
            (
                "ERM RF",
                _run_erm_rf,
                dict(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    E_test=E_test,
                    rep=rep,
                    rep_seed=rep_seed,
                    n_jobs=n_jobs,
                ),
            ),
            (
                "ERM LR",
                _run_erm_lr,
                dict(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    E_test=E_test,
                    rep=rep,
                    rep_seed=rep_seed,
                ),
            ),
        ]

        # ── oracle baselines ──────────────────────────────────────────
        if stable_indices is not None and len(stable_indices) > 0:
            _baseline_tasks.extend(
                [
                    (
                        "Oracle RF",
                        _run_oracle_rf,
                        dict(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            E_test=E_test,
                            stable_indices=stable_indices,
                            rep=rep,
                            rep_seed=rep_seed,
                            n_jobs=n_jobs,
                        ),
                    ),
                    (
                        "Oracle LR",
                        _run_oracle_lr,
                        dict(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            E_test=E_test,
                            stable_indices=stable_indices,
                            rep=rep,
                            rep_seed=rep_seed,
                        ),
                    ),
                ]
            )

        # ── IRM baselines ─────────────────────────────────────────────
        _baseline_tasks.extend(
            [
                (
                    "IRM-NN",
                    _run_irm_nn,
                    dict(
                        X_train=X_train,
                        y_train=y_train,
                        E_train=E_train,
                        X_test=X_test,
                        y_test=y_test,
                        E_test=E_test,
                        rep=rep,
                        rep_seed=rep_seed,
                        n_jobs=n_jobs,
                        verbose=verbose,
                    ),
                ),
                (
                    "IRM-Linear",
                    _run_irm_linear,
                    dict(
                        X_train=X_train,
                        y_train=y_train,
                        E_train=E_train,
                        X_test=X_test,
                        y_test=y_test,
                        E_test=E_test,
                        rep=rep,
                        rep_seed=rep_seed,
                        n_jobs=n_jobs,
                        verbose=verbose,
                    ),
                ),
            ]
        )

        for bname, bfunc, bkwargs in _baseline_tasks:
            t0 = time.time()
            print(f"  Running {bname} ...", end=" ", flush=True)
            rows = bfunc(**bkwargs)
            _append_to_csv(pd.DataFrame(rows), raw_path)
            print(f"done ({time.time() - t0:.1f}s)")

    elapsed_total = time.time() - t_start_total
    print(f"\n{'=' * 70}")
    print(f"All {n_reps} repetitions complete in {elapsed_total / 60:.1f} min")
    print(f"{'=' * 70}")

    # ── compute and save summary ──────────────────────────────────────
    summary_df = compute_summary(raw_path, sc_subsets_path, summary_path)

    # print summary table
    print("\nSUMMARY (mean ± 95% CI)")
    print("─" * 90)
    print(
        f"{'Method':<40s}  {'Min Acc ↑':>18s}  {'Max BCE ↓':>18s}  {'Mean Acc ↑':>18s}"
    )
    print("─" * 90)
    for _, r in summary_df.iterrows():
        min_acc_str = f"{r['min_acc_mean']:.4f}±{r['min_acc_ci']:.4f}"
        max_bce_str = f"{r['max_bce_mean']:.4f}±{r['max_bce_ci']:.4f}"
        mean_acc_str = f"{r['mean_acc_mean']:.4f}±{r['mean_acc_ci']:.4f}"
        print(
            f"  {r['method']:<38s}  {min_acc_str:>18s}  "
            f"{max_bce_str:>18s}  {mean_acc_str:>18s}"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OOD performance evaluation on causal chambers datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="1a",
        choices=["1a", "1b", "2"],
        help="Dataset base name (default: 1a).",
    )
    parser.add_argument(
        "--n-obs",
        type=int,
        default=200,
        help="Number of observations per environment in training subsample (default: 200).",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=20,
        help="Number of repetitions (default: 20).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=10,
        help="Number of parallel workers for internal parallelism (default: 10).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for individual methods.",
    )
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        n_obs_per_env=args.n_obs,
        n_reps=args.n_reps,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
