"""
Adversarial follower experiment for D-spur (lambda=0).

The follower observes W = R = (red, green, blue) and, without observing Y,
chooses a coefficient mechanism theta = (base_led, coef_led, base_pol, coef_pol)
from a finite action menu.  The chamber then sets:

    led_3_ir = base_led + Y * coef_led
    pol_2    = base_pol + Y * coef_pol

and measures X = (R, B, Z) where B = (ir_2, vis_2) and Z = (ir_3, vis_3).

Two predictors are compared:
    f_sb   trained only on R = {red, green, blue}  -- stable blanket
    f_all  trained on all features (R, B, Z)

Action menu (17 actions, strength in {0, 0.5, 1.0}):
    Each action is parameterised by (l0, l1, p0, p1):
        led_3_ir | Y=0 = l0,  led_3_ir | Y=1 = l1
        pol_2    | Y=0 = p0,  pol_2    | Y=1 = p1
    Converted to (base, coef): base_led=l0, coef_led=l1-l0, base_pol=p0, coef_pol=p1-p0.
    The reference action s0_zero has (l0,l1,p0,p1)=(0,0,0,0).
    At strength s: L=round(25*s), P=round(60*s).
    LED patterns: off=(0,0), pos=(0,L), rev=(L,0).
    Pol patterns: off=(0,0), pos=(0,P), rev=(P,0).
    All (off,off) combinations are deduplicated -> 1 + 8 + 8 = 17 actions.

Data collection (two-stage, causal chambers):
    Phase 1:  set RGB -> measure ir_1 -> Y_i = 1{ir_1 > THRESHOLD}
    Phase 2:  for each action theta_m, one experiment with Y-dependent
              led_3_ir / pol_2 -> measures B_i^(m) and Z_i^(m)
    Yields action cube (R_i, Y_i, {B_i^(m), Z_i^(m)}_m) for i=1,...,N.

Follower best-response (probe split, lambda=0):
    For each predictor f, bin t in {0,1,2}, and eligible action m:
        mean_f(t, m) = mean_{i: T_i=t} f(R_i, B_i^(m), Z_i^(m))
    Best response at budget b: pi_f^b(t) = argmin_{m: strength(m)<=b} mean_f(t, m)
    RGB bins are tertiles of f_type(R) = f_sb(R) on the probe set.

Evaluation (eval split):
    For each budget b and predictor f, apply pi_f^b to select (B, Z) per sample.
    Outputs: fixed_action_results.csv, adversarial_results_by_budget.csv,
             adversarial_budget_curves.{png,pdf}
"""

import os
import time

import causalchamber.lab as lab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from utils import sample_truncnorm_integers, wait_for_completion

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 123
N_SAMPLES = 2400  # split equally into probe and eval
THRESHOLD = 12500  # Y = 1{ir_1 > THRESHOLD}, same as D-spur
K_BINS = 3  # tertile bins of f_type score for follower policy
BUDGETS = [0.0, 0.5, 1.0]

SB_FEATURES = ["red", "green", "blue"]
ALL_FEATURES = ["red", "green", "blue", "ir_2", "vis_2", "ir_3", "vis_3"]

PRED_COLORS = {"f_sb": "#0072B2", "f_all": "#D55E00"}
PRED_LABELS = {"f_sb": r"$\hat{f}_{\mathrm{SB}}$", "f_all": r"$\hat{f}_{\mathrm{all}}$"}


# ─── action menu ──────────────────────────────────────────────────────────────


def make_actions(strengths=(0.5, 1.0)):
    """Build the finite action menu.

    Each entry records (base_led, coef_led, base_pol, coef_pol) and the
    (strength, led_pattern, pol_pattern) metadata used for budget filtering.
    """
    actions = [
        {
            "name": "s0_zero",
            "base_led": 0,
            "coef_led": 0,
            "base_pol": 0,
            "coef_pol": 0,
            "strength": 0.0,
            "led_pattern": "off",
            "pol_pattern": "off",
        }
    ]
    seen = {(0, 0, 0, 0)}  # keys are (l0, l1, p0, p1)

    for s in strengths:
        L = int(round(25 * s))
        P = int(round(60 * s))
        led_pairs = {"off": (0, 0), "pos": (0, L), "rev": (L, 0)}
        pol_pairs = {"off": (0, 0), "pos": (0, P), "rev": (P, 0)}

        for led_name, (l0, l1) in led_pairs.items():
            for pol_name, (p0, p1) in pol_pairs.items():
                key = (l0, l1, p0, p1)
                if key in seen:
                    continue
                seen.add(key)
                actions.append(
                    {
                        "name": f"s{s:g}_led_{led_name}_pol_{pol_name}",
                        "base_led": l0,
                        "coef_led": l1 - l0,
                        "base_pol": p0,
                        "coef_pol": p1 - p0,
                        "strength": float(s),
                        "led_pattern": led_name,
                        "pol_pattern": pol_name,
                    }
                )

    return actions


ACTIONS = make_actions(strengths=(0.5, 1.0))
M = len(ACTIONS)
ACTION_NAMES = [a["name"] for a in ACTIONS]
print(f"Action menu: {M} actions  ({[a['name'] for a in ACTIONS]})")


# ─── helpers ──────────────────────────────────────────────────────────────────


def build_X(R, B, Z, features):
    cols = {
        "red": R[:, 0],
        "green": R[:, 1],
        "blue": R[:, 2],
        "ir_2": B[:, 0],
        "vis_2": B[:, 1],
        "ir_3": Z[:, 0],
        "vis_3": Z[:, 1],
    }
    return np.column_stack([cols[f] for f in features])


def eval_metrics(y_true, p, threshold=0.5):
    y1 = y_true == 1
    y0 = y_true == 0
    return {
        "brier": float(np.mean((p - y_true) ** 2)),
        "bce": float(log_loss(y_true, p)),
        "ef": float(p.mean()),
        "ef_y1": float(p[y1].mean()) if y1.any() else float("nan"),
        "ef_y0": float(p[y0].mean()) if y0.any() else float("nan"),
        "fnr": float(np.mean(p[y1] < threshold)) if y1.any() else float("nan"),
    }


# ─── train predictors on existing D-spur training data ────────────────────────

rlab = lab.Lab(os.path.join(DIR, ".credentials"))

df_train = pd.read_csv(os.path.join(DATA_DIR, "d_spur_train.csv"))
X_train_sb = df_train[SB_FEATURES].values
X_train_all = df_train[ALL_FEATURES].values
y_train = df_train["Y"].values.astype(int)

f_sb = RandomForestClassifier(n_estimators=500, random_state=SEED)
f_sb.fit(X_train_sb, y_train)

f_all = RandomForestClassifier(n_estimators=500, random_state=SEED)
f_all.fit(X_train_all, y_train)

print(f"Trained f_sb  on {SB_FEATURES}")
print(f"Trained f_all on {ALL_FEATURES}")

PREDICTORS = {"f_sb": (f_sb, SB_FEATURES), "f_all": (f_all, ALL_FEATURES)}


# ─── data collection ──────────────────────────────────────────────────────────

rgb_inputs = {
    "red": sample_truncnorm_integers(
        N_SAMPLES, mean=64, std=20, low=0, high=255, random_state=SEED + 11
    ),
    "green": sample_truncnorm_integers(
        N_SAMPLES, mean=32, std=30, low=0, high=255, random_state=SEED + 12
    ),
    "blue": sample_truncnorm_integers(
        N_SAMPLES, mean=90, std=12, low=0, high=255, random_state=SEED + 13
    ),
}
R_all = np.column_stack([rgb_inputs["red"], rgb_inputs["green"], rgb_inputs["blue"]])

# phase 1: set RGB, measure ir_1 -> Y
print("\nSubmitting phase 1...")
exp_p1 = rlab.new_experiment(chamber_id="lt-test-b0ni", config="standard")
exp_p1.from_df(pd.DataFrame(rgb_inputs))
eid_p1 = exp_p1.submit(tag="adv_follower_phase1")
time.sleep(2)

print("Waiting for phase 1...")
wait_for_completion(rlab)

df_p1 = rlab.download_data(eid_p1, root=os.path.join(DIR, "tmp")).dataframe
Y_all = np.where(df_p1["ir_1"].values > THRESHOLD, 1, 0)
print(f"Phase 1 complete. N={N_SAMPLES}, P(Y=1)={Y_all.mean():.3f}")

# phase 2: one experiment per action; same RGB, Y-dependent led_3_ir / pol_2
print(f"\nSubmitting {M} phase 2 experiments...")
eids_p2 = {}
for action in ACTIONS:
    inputs_p2 = {k: v.copy() for k, v in rgb_inputs.items()}
    inputs_p2["led_3_ir"] = np.clip(
        action["base_led"] + Y_all * action["coef_led"], 0, None
    ).astype(int)
    inputs_p2["pol_2"] = np.clip(
        action["base_pol"] + Y_all * action["coef_pol"], 0, None
    ).astype(int)

    exp_p2 = rlab.new_experiment(chamber_id="lt-test-b0ni", config="standard")
    exp_p2.from_df(pd.DataFrame(inputs_p2))
    safe_name = action["name"].replace(".", "p")
    eid = exp_p2.submit(tag=f"adv_follower_phase2_{safe_name}")
    eids_p2[action["name"]] = eid
    time.sleep(2)

print("Waiting for phase 2...")
wait_for_completion(rlab)

B_by_action = {}
Z_by_action = {}
for action_name, eid in eids_p2.items():
    df_p2 = rlab.download_data(eid, root=os.path.join(DIR, "tmp")).dataframe
    B_by_action[action_name] = df_p2[["ir_2", "vis_2"]].values
    Z_by_action[action_name] = df_p2[["ir_3", "vis_3"]].values

print(f"Phase 2 complete. Action cube: {N_SAMPLES} units x {M} actions.")

# save action cube
df_cube = pd.DataFrame(
    {
        "Y": Y_all,
        "red": R_all[:, 0],
        "green": R_all[:, 1],
        "blue": R_all[:, 2],
        **{f"ir_2_{n}": B_by_action[n][:, 0] for n in ACTION_NAMES},
        **{f"vis_2_{n}": B_by_action[n][:, 1] for n in ACTION_NAMES},
        **{f"ir_3_{n}": Z_by_action[n][:, 0] for n in ACTION_NAMES},
        **{f"vis_3_{n}": Z_by_action[n][:, 1] for n in ACTION_NAMES},
    }
)
df_cube.to_csv(os.path.join(DATA_DIR, "adversarial_action_cube.csv"), index=False)
print("Saved action cube.")


# ─── probe / eval split ───────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
perm = rng.permutation(N_SAMPLES)
n_probe = N_SAMPLES // 2
probe_idx = perm[:n_probe]
eval_idx = perm[n_probe:]


# ─── RGB binning: tertiles of f_type = f_sb on the probe set ─────────────────

# f_type defines the typing rule T(R); it is distinct from the deployed predictor.
f_type = f_sb
R_probe = R_all[probe_idx]
type_scores_probe = f_type.predict_proba(R_probe)[:, 1]
bin_edges = np.percentile(
    type_scores_probe, [100 * k / K_BINS for k in range(1, K_BINS)]
)


def assign_bins(scores):
    """Map scores to bins {0, ..., K_BINS-1} using probe-derived tertile edges."""
    return np.digitize(scores, bin_edges)


T_probe = assign_bins(type_scores_probe)
T_eval = assign_bins(f_type.predict_proba(R_all[eval_idx])[:, 1])

print(
    f"\nProbe bin counts (K={K_BINS}): "
    f"{ {t: int((T_probe == t).sum()) for t in range(K_BINS)} }"
)


# ─── probe mean scores for all predictors x all actions ──────────────────────

# probe_mean_score[pred_name] has shape (K_BINS, M):
#   probe_mean_score[pred_name][t, m] = mean E[f(X^m) | T=t] on probe split.

print("\n─── Computing probe mean scores ───")
probe_mean_score = {}
for pred_name, (predictor, features) in PREDICTORS.items():
    ms = np.full((K_BINS, M), np.nan)
    for m, action_name in enumerate(ACTION_NAMES):
        X_m = build_X(
            R_probe,
            B_by_action[action_name][probe_idx],
            Z_by_action[action_name][probe_idx],
            features,
        )
        scores = predictor.predict_proba(X_m)[:, 1]
        for t in range(K_BINS):
            mask = T_probe == t
            if mask.sum() > 0:
                ms[t, m] = scores[mask].mean()
    probe_mean_score[pred_name] = ms
    print(f"  {pred_name}: done")


# ─── fixed-action diagnostics (eval split) ───────────────────────────────────

print("\n─── Fixed-action diagnostics (eval split) ───")
print(
    f"{'predictor':<10}  {'action':<32}  {'Brier':>8}  {'BCE':>8}  "
    f"{'E[f]':>8}  {'E[f|Y=1]':>10}  {'E[f|Y=0]':>10}  {'FNR':>8}"
)
print("─" * 100)

R_eval = R_all[eval_idx]
Y_eval = Y_all[eval_idx]

fixed_rows = []
for pred_name, (predictor, features) in PREDICTORS.items():
    for action_name in ACTION_NAMES:
        X_m = build_X(
            R_eval,
            B_by_action[action_name][eval_idx],
            Z_by_action[action_name][eval_idx],
            features,
        )
        p_m = predictor.predict_proba(X_m)[:, 1]
        met = eval_metrics(Y_eval, p_m)
        print(
            f"{pred_name:<10}  {action_name:<32}  {met['brier']:>8.4f}  {met['bce']:>8.4f}  "
            f"{met['ef']:>8.4f}  {met['ef_y1']:>10.4f}  {met['ef_y0']:>10.4f}  {met['fnr']:>8.4f}"
        )
        fixed_rows.append({"predictor": pred_name, "action": action_name, **met})
    print()

pd.DataFrame(fixed_rows).to_csv(
    os.path.join(DATA_DIR, "fixed_action_results.csv"), index=False
)
print("Saved fixed_action_results.csv")


# ─── best-response by budget (eval split) ────────────────────────────────────

print("\n─── Best-response by budget (eval split) ───")
print(
    f"{'pred':<10}  {'budget':>7}  {'clean Brier':>12}  {'adv Brier':>12}  "
    f"{'clean E[f]':>12}  {'adv E[f]':>12}  {'adv FNR':>10}"
)
print("─" * 90)

THETA0 = ACTIONS[0]["name"]  # s0_zero — used as the clean reference
result_rows = []

for budget in BUDGETS:
    eligible_idx = [i for i, a in enumerate(ACTIONS) if a["strength"] <= budget]

    for pred_name, (predictor, features) in PREDICTORS.items():
        ms_eligible = probe_mean_score[pred_name][:, eligible_idx]
        best_local = np.argmin(ms_eligible, axis=1)  # index within eligible
        best_m_per_bin = np.array(eligible_idx)[best_local]  # global action index

        # clean: always under theta_0 reference
        X_clean = build_X(
            R_eval,
            B_by_action[THETA0][eval_idx],
            Z_by_action[THETA0][eval_idx],
            features,
        )
        p_clean = predictor.predict_proba(X_clean)[:, 1]
        met_clean = eval_metrics(Y_eval, p_clean)

        # adversarial: per-bin best response selects both B and Z
        B_adv = np.empty((len(eval_idx), 2))
        Z_adv = np.empty((len(eval_idx), 2))
        for t in range(K_BINS):
            mask = T_eval == t
            aname = ACTION_NAMES[best_m_per_bin[t]]
            B_adv[mask] = B_by_action[aname][eval_idx][mask]
            Z_adv[mask] = Z_by_action[aname][eval_idx][mask]

        X_adv = build_X(R_eval, B_adv, Z_adv, features)
        p_adv = predictor.predict_proba(X_adv)[:, 1]
        met_adv = eval_metrics(Y_eval, p_adv)

        selected = [ACTION_NAMES[best_m_per_bin[t]] for t in range(K_BINS)]
        print(
            f"{pred_name:<10}  {budget:>7.1f}  {met_clean['brier']:>12.4f}  {met_adv['brier']:>12.4f}  "
            f"{met_clean['ef']:>12.4f}  {met_adv['ef']:>12.4f}  {met_adv['fnr']:>10.4f}"
        )
        print(f"    selected actions: {selected}")
        result_rows.append(
            {
                "predictor": pred_name,
                "budget": budget,
                **{f"clean_{k}": v for k, v in met_clean.items()},
                **{f"adv_{k}": v for k, v in met_adv.items()},
                **{
                    f"best_action_bin{t}": ACTION_NAMES[best_m_per_bin[t]]
                    for t in range(K_BINS)
                },
            }
        )

df_budget = pd.DataFrame(result_rows)
df_budget.to_csv(
    os.path.join(DATA_DIR, "adversarial_results_by_budget.csv"), index=False
)
print("\nSaved adversarial_results_by_budget.csv")


# ─── plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

for pred_name in PREDICTORS:
    sub = df_budget[df_budget["predictor"] == pred_name].sort_values("budget")
    color = PRED_COLORS[pred_name]
    label = PRED_LABELS[pred_name]
    axes[0].plot(sub["budget"], sub["adv_ef"], marker="o", color=color, label=label)
    axes[1].plot(sub["budget"], sub["adv_brier"], marker="o", color=color, label=label)
    # clean reference (constant across budget) as dashed line
    axes[0].axhline(sub["clean_ef"].iloc[0], color=color, linestyle="--", alpha=0.35)
    axes[1].axhline(sub["clean_brier"].iloc[0], color=color, linestyle="--", alpha=0.35)

axes[0].set_xlabel("follower budget")
axes[0].set_ylabel(r"$\mathbb{E}_{\hat\pi}[f(X)]$")
axes[0].legend()
axes[1].set_xlabel("follower budget")
axes[1].set_ylabel(r"Brier score under $\hat\pi$")
axes[1].legend()
plt.tight_layout()

plot_base = os.path.join(DATA_DIR, "adversarial_budget_curves")
plt.savefig(plot_base + ".png", dpi=150)
plt.savefig(plot_base + ".pdf")
plt.close()
print(f"Saved plot to {plot_base}.{{png,pdf}}")
