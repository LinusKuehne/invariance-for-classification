"""
GCM conditional independence tests: sensor_i ⊥ sensor_j | (red, green, blue)
Run for environments E=0,1,2 in d_spur_train.csv.

Pairs tested:
  ir_1  ⊥ ir_2,   ir_2  ⊥ ir_3
  vis_1 ⊥ vis_2,  vis_2 ⊥ vis_3

Uses weightedGCM::wgcm.est via rpy2 — a generalisation of the
Generalised Covariance Measure of Shah & Peters (2020).
Returns a p-value for H0: X ⊥ Y | (red, green, blue).
"""

import os

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

wGCM = importr("weightedGCM")
_conv = ro.default_converter + numpy2ri.converter

SEED = 1
np.random.seed(SEED)

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")

COND = ["red", "green", "blue"]
ENVS = [0, 1, 2]
PAIRS = [("ir_1", "ir_2"), ("ir_2", "ir_3"), ("vis_1", "vis_2"), ("vis_2", "vis_3")]

df = pd.read_csv(os.path.join(DATA_DIR, "d_spur_train.csv"))


def wgcm_p(sub: pd.DataFrame, x_var: str, y_var: str) -> float:
    X = np.asarray(sub[[x_var]], dtype=float)
    Y = np.asarray(sub[[y_var]], dtype=float)
    Z = np.asarray(sub[COND], dtype=float)
    ro.r(f"set.seed({SEED})")
    with localconverter(_conv):
        return float(wGCM.wgcm_est(X, Y, Z, regr_meth="xgboost")[0])


results = {}
for x_var, y_var in PAIRS:
    for e in ENVS:
        results[(x_var, y_var, e)] = wgcm_p(df[df["E"] == e], x_var, y_var)
    results[(x_var, y_var, "pooled")] = wgcm_p(df[df["E"].isin(ENVS)], x_var, y_var)


def fmt_p(p: float) -> str:
    return r"$<0.001$" if p < 0.001 else f"${p:.3f}$"


def latex_var(name: str) -> str:
    return rf"$\mathtt{{{name.replace('_', r'\_')}}}$"


print(r"\begin{tabular}{llrrrr}")
print(r"\toprule")
print(r" & & \multicolumn{4}{c}{$p$-value} \\")
print(r"\cmidrule(lr){3-6}")
print(r"$Z_1$ & $Z_2$ & $E=0$ & $E=1$ & $E=2$ & $E \in \{0,1,2\}$ \\")
print(r"\midrule")
for x_var, y_var in PAIRS:
    px = latex_var(x_var)
    py = latex_var(y_var)
    ps = " & ".join(fmt_p(results[(x_var, y_var, e)]) for e in ENVS)
    pp = fmt_p(results[(x_var, y_var, "pooled")])
    print(rf"{px} & {py} & {ps} & {pp} \\")
print(r"\bottomrule")
print(r"\end{tabular}")
