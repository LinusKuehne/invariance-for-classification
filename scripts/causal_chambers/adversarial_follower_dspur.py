"""
Adversarial follower experiment for D-spur (lambda=0).

The follower observes W = R = (red, green, blue) and, without observing Y,
chooses a coefficient mechanism theta = (base_led, coef_led, base_pol, coef_pol)
from a finite action menu.  The chamber then sets:

    led_3_ir = base_led + Y * coef_led
    pol_2    = base_pol + Y * coef_pol

and measures Z = (ir_3, vis_3).  The feature vector is X = (R, B, Z) where
B = (ir_2, vis_2) is action-independent (measured in phase 1 before any action
is committed).

Two predictors are compared:
    f_sb   trained only on R = {red, green, blue}  -- stable blanket
    f_all  trained on all features (R, B, Z)

Action menu:
    theta_0     (base_led=0,  coef_led=0,   base_pol=0,  coef_pol=0)   reference
    theta_plus  (base_led=0,  coef_led=25,  base_pol=0,  coef_pol=60)  training-like
    theta_minus (base_led=25, coef_led=-25, base_pol=60, coef_pol=-60) reversed (failure case)

Data collection (two-stage, causal chambers):
    Phase 1: set RGB -> measure ir_1 -> Y_i = 1{ir_1 > THRESHOLD}; also read B_i
    Phase 2: for each action theta_m, one experiment with Y-dependent led_3_ir / pol_2
             -> measures Z_i^(m) = (ir_3^(m), vis_3^(m))
    Yields action cube (R_i, B_i, Y_i, Z_i^(0), Z_i^(+), Z_i^(-)) for i=1,...,N.

Follower best-response (probe split, lambda=0):
    For each predictor f, bin t in {0,1,2}, and action m:
        mean_f(t, m) = mean_{i: T_i=t} f(R_i, B_i, Z_i^(m))
    Best response: pi_f(t) = argmin_m mean_f(t, m)
    RGB bins are tertiles of f_sb(R) on the probe set.

Evaluation (eval split):
    For each i: select Z_i^(pi_f(T_i)), construct X_i = (R_i, B_i, Z_i^(pi_f(T_i))).
    Report Brier score, BCE, and E[f(X)] under clean (theta_0) and adversarial policy.
"""

import os
import time

import causalchamber.lab as lab
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from utils import sample_truncnorm_integers, wait_for_completion

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 123
N_SAMPLES = 1200  # split equally into probe and eval
THRESHOLD = 12500  # Y = 1{ir_1 > THRESHOLD}, same as D-spur
K_BINS = 3  # tertile bins of f_sb score for follower policy

SB_FEATURES = ["red", "green", "blue"]
ALL_FEATURES = ["red", "green", "blue", "ir_2", "vis_2", "ir_3", "vis_3"]

# Finite action menu for the follower.
# theta_minus is the known failure case from test envs 0 and 3 of D-spur.
ACTIONS = [
    {"name": "theta_0", "base_led": 0, "coef_led": 0, "base_pol": 0, "coef_pol": 0},
    {
        "name": "theta_plus",
        "base_led": 0,
        "coef_led": 25,
        "base_pol": 0,
        "coef_pol": 60,
    },
    {
        "name": "theta_minus",
        "base_led": 25,
        "coef_led": -25,
        "base_pol": 60,
        "coef_pol": -60,
    },
]
M = len(ACTIONS)
ACTION_NAMES = [a["name"] for a in ACTIONS]

rlab = lab.Lab(os.path.join(DIR, ".credentials"))


# ─── train predictors on existing D-spur training data ────────────────────────

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


# ─── data collection ──────────────────────────────────────────────────────────

# draw reference RGB for all N_SAMPLES observations -- fixed across all actions
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

# phase 1: set RGB, measure ir_1 (-> Y) and B = (ir_2, vis_2)
# B is read here, before any action is committed, so it is action-independent.
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
# -> each yields Z^(m) = (ir_3, vis_3) for that action
print("\nSubmitting phase 2 experiments (one per action)...")
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
    eid = exp_p2.submit(tag=f"adv_follower_phase2_{action['name']}")
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

# save action cube for reproducibility
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
cube_path = os.path.join(DATA_DIR, "adversarial_action_cube.csv")
df_cube.to_csv(cube_path, index=False)
print(f"Saved action cube to {cube_path}")


# ─── probe / eval split ───────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
perm = rng.permutation(N_SAMPLES)
n_probe = N_SAMPLES // 2
probe_idx = perm[:n_probe]
eval_idx = perm[n_probe:]


# ─── RGB binning: tertiles of f_sb score on probe set ────────────────────────

R_probe = R_all[probe_idx]
sb_scores_probe = f_sb.predict_proba(R_probe)[:, 1]
bin_edges = np.percentile(sb_scores_probe, [100 * k / K_BINS for k in range(1, K_BINS)])


def assign_bins(scores):
    """Map scores to bins {0, ..., K_BINS-1} using probe-derived tertile edges."""
    return np.digitize(scores, bin_edges)


T_probe = assign_bins(sb_scores_probe)
T_eval = assign_bins(f_sb.predict_proba(R_all[eval_idx])[:, 1])

print(
    f"\nProbe bin counts (K={K_BINS}): "
    f"{ {t: int((T_probe == t).sum()) for t in range(K_BINS)} }"
)


# ─── best-response computation (probe split) ─────────────────────────────────


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


PREDICTORS = {
    "f_sb": (f_sb, SB_FEATURES),
    "f_all": (f_all, ALL_FEATURES),
}

best_response = {}  # pred_name -> int array of shape (K_BINS,), action indices

print("\n─── Best-response computation (probe split) ───")
for pred_name, (predictor, features) in PREDICTORS.items():
    # mean_score[t, m] = mean predicted P(Y=1) for bin t under action m
    mean_score = np.full((K_BINS, M), np.nan)

    for m, action_name in enumerate(ACTION_NAMES):
        B_probe_m = B_by_action[action_name][probe_idx]
        Z_probe_m = Z_by_action[action_name][probe_idx]
        X_probe_m = build_X(R_probe, B_probe_m, Z_probe_m, features)
        scores = predictor.predict_proba(X_probe_m)[:, 1]
        for t in range(K_BINS):
            mask = T_probe == t
            if mask.sum() > 0:
                mean_score[t, m] = scores[mask].mean()

    # follower minimizes E[f(X)] -> argmin over actions per bin
    best_m_per_bin = np.argmin(mean_score, axis=1)
    best_response[pred_name] = best_m_per_bin

    print(f"\n  {pred_name}:")
    print(
        f"    {'bin':>4}  " + "  ".join(f"{n:>14}" for n in ACTION_NAMES) + "  -> best"
    )
    for t in range(K_BINS):
        vals = "  ".join(f"{mean_score[t, m]:>14.4f}" for m in range(M))
        print(f"    {t:>4}  {vals}  -> {ACTION_NAMES[best_m_per_bin[t]]}")


# ─── evaluation on eval split ─────────────────────────────────────────────────

R_eval = R_all[eval_idx]
Y_eval = Y_all[eval_idx]

# ── fixed-action diagnostics: performance under each action uniformly applied ──

print("\n─── Fixed-action diagnostics (eval split) ───")
print(f"{'predictor':<10}  {'action':<14}  {'Brier':>10}  {'BCE':>10}  {'E[f]':>10}")
print("─" * 60)

for pred_name, (predictor, features) in PREDICTORS.items():
    for action_name in ACTION_NAMES:
        X_m = build_X(
            R_eval,
            B_by_action[action_name][eval_idx],
            Z_by_action[action_name][eval_idx],
            features,
        )
        p_m = predictor.predict_proba(X_m)[:, 1]
        brier = float(np.mean((p_m - Y_eval) ** 2))
        bce = float(log_loss(Y_eval, p_m))
        ef = float(p_m.mean())
        print(
            f"{pred_name:<10}  {action_name:<14}  {brier:>10.4f}  {bce:>10.4f}  {ef:>10.4f}"
        )
    print()

# ── main result: clean vs best-response ───────────────────────────────────────

print("\n─── Best-response evaluation (eval split) ───")
print(
    f"{'predictor':<10}  {'clean Brier':>12}  {'adv Brier':>12}  "
    f"{'clean BCE':>12}  {'adv BCE':>12}  {'clean E[f]':>12}  {'adv E[f]':>12}"
)
print("─" * 90)

result_rows = []
for pred_name, (predictor, features) in PREDICTORS.items():
    # clean: evaluate under theta_0 (reference, no spurious shift)
    X_clean = build_X(
        R_eval,
        B_by_action["theta_0"][eval_idx],
        Z_by_action["theta_0"][eval_idx],
        features,
    )
    p_clean = predictor.predict_proba(X_clean)[:, 1]

    # adversarial: apply per-bin best-response policy, selecting both B and Z
    B_adv = np.empty((len(eval_idx), 2))
    Z_adv = np.empty((len(eval_idx), 2))
    best_m_per_bin = best_response[pred_name]
    for t in range(K_BINS):
        mask = T_eval == t
        action_name = ACTION_NAMES[best_m_per_bin[t]]
        B_adv[mask] = B_by_action[action_name][eval_idx][mask]
        Z_adv[mask] = Z_by_action[action_name][eval_idx][mask]

    X_adv = build_X(R_eval, B_adv, Z_adv, features)
    p_adv = predictor.predict_proba(X_adv)[:, 1]

    brier_clean = float(np.mean((p_clean - Y_eval) ** 2))
    brier_adv = float(np.mean((p_adv - Y_eval) ** 2))
    bce_clean = float(log_loss(Y_eval, p_clean))
    bce_adv = float(log_loss(Y_eval, p_adv))
    ef_clean = float(p_clean.mean())
    ef_adv = float(p_adv.mean())

    print(
        f"{pred_name:<10}  {brier_clean:>12.4f}  {brier_adv:>12.4f}  "
        f"{bce_clean:>12.4f}  {bce_adv:>12.4f}  {ef_clean:>12.4f}  {ef_adv:>12.4f}"
    )
    result_rows.append(
        {
            "predictor": pred_name,
            "clean_brier": brier_clean,
            "adv_brier": brier_adv,
            "clean_bce": bce_clean,
            "adv_bce": bce_adv,
            "clean_ef": ef_clean,
            "adv_ef": ef_adv,
            **{
                f"best_action_bin{t}": ACTION_NAMES[best_m_per_bin[t]]
                for t in range(K_BINS)
            },
        }
    )

results_path = os.path.join(DATA_DIR, "adversarial_results.csv")
pd.DataFrame(result_rows).to_csv(results_path, index=False)
print(f"\nSaved results to {results_path}")
