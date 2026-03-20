"""
Verify hypothesis: subsets including ir_3 or vis_3 only rarely get p-value > 0.05
for TramGCM(LR) and TramGCM(RF) on dataset 2.
"""

import json
import os

import numpy as np
import pandas as pd

ALPHA = 0.05
TESTS = ["TramGCM(LR)", "TramGCM(RF)"]

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pvalues_path = os.path.join(repo_root, "results", "2", "OOD_sc_pvalues_2_n200.csv")

df = pd.read_csv(pvalues_path)
df_tramgcm = df[df["sc_config"].isin(TESTS)].copy()

# parse subset column
df_tramgcm["subset_set"] = df_tramgcm["subset"].apply(
    lambda s: frozenset(json.loads(s))
)
df_tramgcm["has_ir3"] = df_tramgcm["subset_set"].apply(lambda s: "ir_3" in s)
df_tramgcm["has_vis3"] = df_tramgcm["subset_set"].apply(lambda s: "vis_3" in s)
df_tramgcm["has_ir3_or_vis3"] = df_tramgcm["has_ir3"] | df_tramgcm["has_vis3"]
df_tramgcm["above_alpha"] = df_tramgcm["p_value_or_score"] >= ALPHA

n_reps = df_tramgcm["rep"].nunique()
print(f"Repetitions: {n_reps}")
print(f"Alpha: {ALPHA}")
print()

for test in TESTS:
    sub = df_tramgcm[df_tramgcm["sc_config"] == test]
    n_subsets_per_rep = sub.groupby("rep")["subset"].nunique().mean()

    print("=" * 60)
    print(f"Test: {test}  (n_subsets/rep ≈ {n_subsets_per_rep:.0f})")
    print("=" * 60)

    for label, mask_col in [
        ("subsets WITHOUT ir_3/vis_3", ~sub["has_ir3_or_vis3"]),
        ("subsets WITH ir_3 or vis_3", sub["has_ir3_or_vis3"]),
        ("  subsets WITH ir_3 only", sub["has_ir3"] & ~sub["has_vis3"]),
        ("  subsets WITH vis_3 only", sub["has_vis3"] & ~sub["has_ir3"]),
        ("  subsets WITH both ir_3 and vis_3", sub["has_ir3"] & sub["has_vis3"]),
    ]:
        group = sub[mask_col]
        n_total = len(group)
        n_above = (group["p_value_or_score"] >= ALPHA).sum()
        pct = 100.0 * n_above / n_total if n_total > 0 else float("nan")
        print(
            f"  {label:<42s}  {n_above:>5d}/{n_total:<5d} ({pct:5.1f}%) above {ALPHA}"
        )

    # per-rep counts: how often does *any* ir3/vis3 subset pass alpha?
    reps_with_any_pass = (
        sub[sub["has_ir3_or_vis3"]].groupby("rep")["above_alpha"].any().sum()
    )
    print(
        f"\n  Reps where ≥1 ir_3/vis_3 subset passes alpha: "
        f"{reps_with_any_pass}/{n_reps}  "
        f"({100.0 * reps_with_any_pass / n_reps:.0f}%)"
    )

    # distribution of p-values for ir3/vis3 subsets
    pvals = sub[sub["has_ir3_or_vis3"]]["p_value_or_score"].dropna().to_numpy()
    if len(pvals) > 0:
        print(
            f"  p-value stats for ir_3/vis_3 subsets: "
            f"median={np.median(pvals):.2e}, "
            f"max={np.max(pvals):.4f}, "
            f"mean={np.mean(pvals):.2e}"
        )

    # same stats for clean subsets as reference
    pvals_clean = sub[~sub["has_ir3_or_vis3"]]["p_value_or_score"].dropna().to_numpy()
    if len(pvals_clean) > 0:
        print(
            f"  p-value stats for other subsets:       "
            f"median={np.median(pvals_clean):.2e}, "
            f"max={np.max(pvals_clean):.4f}, "
            f"mean={np.mean(pvals_clean):.2e}"
        )
    print()
