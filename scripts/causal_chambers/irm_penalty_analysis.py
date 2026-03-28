"""
IRM penalty analysis for dataset 2.

Compute the IRMv1 penalty at the ERM (pooled logistic regression) optimum,
following the original IRMv1 formulation

Penalty(w, Phi) = sum_e || grad_{w|w=1} R_e(w * Phi(x)) ||^2,

where R_e is the binary cross-entropy in environment e and Phi(x) is the
representation (here: the ERM logistic model's linear combination of
features, before applying the sigmoid). At w=1 this reduces to:

Penalty_e = ( E_e[ (y - sigmoid(logit)) * logit ] )^2,

where logit = W^T x is the model's output.

We evaluate the penalty at the ERM (pooled) optimum, not at
an IRM-regularised solution to reveal the incentive structure for IRM.

Incremental addition to the stable blanket:
    Start from the stable blanket S = {red, green, blue} and consider the
    sets S, S + {ir_3}, S + {vis_3}.
    Answers: "What tradeoff does IRM face when deciding to include ir_3 (or
    vis_3)?" The penalty increase from adding a feature, relative to the
    training-loss decrease, directly shows whether IRM's soft regularizer
    can resist including the feature.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")

# ── feature configuration ────────────────────────────────────────────────────

ALL_FEATURES = ["red", "green", "blue", "ir_2", "vis_2", "ir_3", "vis_3"]
STABLE_BLANKET = ["red", "green", "blue"]


# ── data loading ──────────────────────────────────────────────────────────────


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(os.path.join(DATA_DIR, "2_train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "2_test.csv"))
    return df_train, df_test


# ── IRM penalty computation ───────────────────────────────────────────────────


def irm_penalty_at_erm_optimum(
    feature_names: list[str],
    df_train: pd.DataFrame,
) -> dict:
    """
    Fit a pooled ERM logistic regression on `feature_names` using all
    training environments, then compute the IRMv1 penalty at that ERM
    optimum.

    Returns a dict with:
      - 'train_bce'         : mean training BCE (lower = better fit)
      - 'penalty_per_env'   : list of per-environment penalties
      - 'total_penalty'     : sum of per-environment penalties
      - 'max_env_penalty'   : max single-environment penalty
    """
    envs = sorted(df_train["E"].unique())

    # ── 1.  Fit pooled ERM ──────────────────────────────────────────────────
    X_all = df_train[feature_names].values.astype(np.float64)
    y_all = df_train["Y"].to_numpy().astype(np.float64)

    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)

    lr = LogisticRegression(
        C=1e6, fit_intercept=True, max_iter=10_000, solver="lbfgs", random_state=0
    )
    lr.fit(X_all_s, y_all)

    p_all = lr.predict_proba(X_all_s)[:, 1]
    train_bce = float(log_loss(y_all, p_all))

    # ── 2.  Compute IRMv1 penalty per environment ───────────────────────────
    # logit_i = w^T x_i + b  (scalar per observation)
    # gradient of env-e BCE w.r.t. dummy scale at scale=1:
    #   g_e = mean_e [ (y_e - sigma(logit_e)) * logit_e ]
    # penalty_e = g_e^2
    w = torch.tensor(lr.coef_[0], dtype=torch.float64)
    b = torch.tensor(lr.intercept_[0], dtype=torch.float64)

    penalty_per_env: list[float] = []
    for e in envs:
        mask = df_train["E"] == e
        X_e = torch.tensor(X_all_s[mask], dtype=torch.float64)
        y_e = torch.tensor(y_all[mask], dtype=torch.float64)

        scale = torch.ones(1, dtype=torch.float64, requires_grad=True)
        logits = X_e @ w + b  # shape (n_e,)
        loss_e = F.binary_cross_entropy_with_logits(logits * scale, y_e)
        (g,) = torch.autograd.grad(loss_e, scale, create_graph=False)
        penalty_per_env.append(float(g) ** 2)

    total_penalty = sum(penalty_per_env)

    return {
        "train_bce": train_bce,
        "penalty_per_env": penalty_per_env,
        "total_penalty": total_penalty,
        "max_env_penalty": max(penalty_per_env),
    }


def analysis_incremental(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    feature_sets: list[tuple[str, list[str], str]] = [
        ("S  (stable blanket only)", STABLE_BLANKET, r"$\SB(Y)$"),
        ("S + ir_3", STABLE_BLANKET + ["ir_3"], r"$\SB(Y) \cup \{\mathtt{ir\_3}\}$"),
        ("S + vis_3", STABLE_BLANKET + ["vis_3"], r"$\SB(Y) \cup \{\mathtt{vis\_3}\}$"),
    ]

    results = []
    for name, feats, tex_name in feature_sets:
        res = irm_penalty_at_erm_optimum(feats, df_train)
        results.append(
            {
                "name": name,
                "tex_name": tex_name,
                "train_bce": res["train_bce"],
                "total_penalty": res["total_penalty"],
            }
        )

    # ── print LaTeX table ─────────────────────────────────────────────────────
    print(r"\begin{tabular}{@{}lccc@{}}")
    print(r"    \toprule")
    print(r"    Feature set")
    print(r"      & $R_\mathrm{tr}$")
    print(r"      & $\mathrm{pen}$")
    print(r"      & Obj.\ \eqref{eq:IRMv1} ($\lambda=0.9$) \\")
    print(r"    \midrule")

    def fmt_sci(x: float) -> str:
        s = f"{x:.3e}"
        mantissa, exp = s.split("e")
        exp_int = int(exp)
        return rf"${mantissa} \times 10^{{{exp_int}}}$"

    def fmt_num(x: float) -> str:
        return f"${x:.3f}$"

    for r in results:
        obj_09 = 0.1 * r["train_bce"] + 0.9 * r["total_penalty"]
        print(
            f"    {r['tex_name']} "
            f"& {fmt_num(r['train_bce'])} "
            f"& {fmt_sci(r['total_penalty'])} "
            f"& {fmt_num(obj_09)} \\\\"
        )

    print(r"    \bottomrule")
    print(r"  \end{tabular}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    df_train, df_test = load_data()

    print()
    print("IRM PENALTY ANALYSIS")
    print()
    print(
        f"Training data: {len(df_train)} samples, "
        f"{df_train['E'].nunique()} environments "
        f"(E={sorted(df_train['E'].unique())})"
    )
    print(
        f"Test data:     {len(df_test)} samples, "
        f"{df_test['E'].nunique()} environments "
        f"(E={sorted(df_test['E'].unique())})"
    )
    print(f"Features:      {ALL_FEATURES}")
    print(f"Stable blanket:{STABLE_BLANKET}")
    print()

    print("── Incremental addition to stable blanket ───────────────────────────────")
    print()
    analysis_incremental(df_train, df_test)


if __name__ == "__main__":
    main()
