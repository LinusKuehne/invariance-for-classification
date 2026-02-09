"""
Dataset 5
---------
Two-stage data generation process with feedback loop.
1. Generate R/G/B -> Measure ir_1 -> Compute Y
2. Set led_3_ir, pol_2 based on Y -> Measure everything
"""

import time

import causalchamber.lab as lab
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.ensemble import RandomForestClassifier

rlab = lab.Lab(".credentials")


DIR = "/Users/linuskuehne/git/invariance-for-classification/scripts/causal_chambers/"


def sample_truncnorm_integers(n, mean, std, low, high, random_state=None):
    a = (low - mean) / std
    b = (high - mean) / std
    values = truncnorm.rvs(a, b, loc=mean, scale=std, size=n, random_state=random_state)
    return np.round(values).astype(int)


def produce_dataset(dataset_type="train"):
    if dataset_type == "train":
        seed = SEED
    else:
        seed = SEED_TEST

    interventions = []
    env_indices = []

    for i, intervention_dict in enumerate(all_interventions[dataset_type]):
        env_indices.append(i)
        interventions.append(intervention_dict.copy())

    # --- PHASE 1: Prior Experiments ---
    experiment_ids_phase1 = []
    phase1_inputs_list = []

    print(f"Submitting Phase 1 experiments ({dataset_type})...")
    for i, intervention in enumerate(interventions):
        # Base inputs
        inputs = reference_setting(seed + i)

        # Apply interventions (excluding coefs which are used in phase 2)
        for target, values in intervention.items():
            if target not in ["coef_led", "coef_pol"]:
                inputs[target] = values

        phase1_inputs_list.append(inputs)

        experiment = rlab.new_experiment(chamber_id="lt-test-b0ni", config="standard")
        experiment.from_df(pd.DataFrame(inputs))
        experiment_id = experiment.submit(
            tag=f"{dataset_type}_env_{env_indices[i]}_prior"
        )
        experiment_ids_phase1.append(experiment_id)

    # Wait for Phase 1
    print(f"Waiting for Phase 1 ({dataset_type})...")
    wait_for_completion()

    # --- PHASE 2: Feedback & Post Experiments ---
    experiment_ids_phase2 = []
    Ys_list = []

    print(f"Submitting Phase 2 experiments ({dataset_type})...")
    for i, prior_eid in enumerate(experiment_ids_phase1):
        # Download Phase 1 results
        try:
            df_prior = rlab.download_data(prior_eid, root=DIR + "tmp").dataframe
        except Exception as e:
            print(f"Error downloading experiment {prior_eid}: {e}")
            raise

        ir_1_prior = df_prior["ir_1"].to_numpy()

        # Compute Y
        threshold = 12500
        Y = np.where(ir_1_prior > threshold, 1, 0)
        Ys_list.append(Y)

        # Determine coefficients (default to 0 if not specified)
        intervention = interventions[i]
        coef_led = intervention.get("coef_led", 0)
        coef_pol = intervention.get("coef_pol", 0)

        # Prepare Phase 2 inputs
        inputs_phase2 = phase1_inputs_list[i].copy()

        # Add feedback interventions
        inputs_phase2["led_3_ir"] = Y * coef_led
        inputs_phase2["pol_2"] = Y * coef_pol

        # Submit Phase 2
        experiment = rlab.new_experiment(chamber_id="lt-test-b0ni", config="standard")
        experiment.from_df(pd.DataFrame(inputs_phase2))
        experiment_id = experiment.submit(
            tag=f"{dataset_type}_env_{env_indices[i]}_post"
        )
        experiment_ids_phase2.append(experiment_id)

    # Wait for Phase 2
    print(f"Waiting for Phase 2 ({dataset_type})...")
    wait_for_completion()

    # --- Collect Data ---
    dataframes = []
    for i, eid in enumerate(experiment_ids_phase2):
        df_post = rlab.download_data(eid, root=DIR + "tmp").dataframe

        # Attach Y and Environment Index
        df_post = df_post.assign(Y=Ys_list[i])
        df_post = df_post.assign(E=env_indices[i])

        dataframes.append(df_post)

    df = pd.concat(dataframes, ignore_index=True)

    # Save datasets
    # Small version
    df[["Y", "red", "green", "blue", "ir_3", "vis_3", "E"]].to_csv(
        DIR + f"{dataset_name}" + f"_small_{dataset_type}.csv", index=False
    )

    # Full version
    df[
        ["Y", "red", "green", "blue", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3", "E"]
    ].to_csv(DIR + f"{dataset_name}" + f"_{dataset_type}.csv", index=False)

    # OOB Score on env 0
    df_test = df[
        ["Y", "red", "green", "blue", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3", "E"]
    ]

    X_train = df_test.drop(columns=["Y"])
    y_train = df_test["Y"]

    rf = RandomForestClassifier(random_state=SEED, bootstrap=True, oob_score=True)
    rf.fit(X_train, y_train)
    print(f"OOB Score {dataset_type}: {rf.oob_score_}")


def wait_for_completion():
    status = rlab.get_experiments(verbose=False)[0]["status"]
    while not status == "DONE":
        time.sleep(15)
        status = rlab.get_experiments(verbose=False)[0]["status"]


N = 200
SEED = 42
SEED_TEST = SEED + 1000


# Reference setting function
def reference_setting(random_state):
    inputs = {
        "red": sample_truncnorm_integers(
            N,
            mean=64,
            std=20,
            low=0,
            high=255,
            random_state=random_state + 11,
        ),
        "green": sample_truncnorm_integers(
            N,
            mean=32,
            std=30,
            low=0,
            high=255,
            random_state=random_state + 12,
        ),
        "blue": sample_truncnorm_integers(
            N,
            mean=90,
            std=12,
            low=0,
            high=255,
            random_state=random_state + 13,
        ),
    }

    return inputs


# Interventions
# Key design principles:
#   - Every environment should have a non-degenerate class balance (avoid Y≈100%)
#   - Vary feedback coefficients across environments so the invariance test
#     has power to detect that descendants (ir_3, vis_3, …) are unstable
#   - Combine color shifts with different feedback coefficients
train_interventions = [
    # Env 0: Reference (no feedback, reference colors)
    {},
    # Env 1: Strong feedback, reference colors
    {"coef_led": 20, "coef_pol": 50},
    # Env 2: Moderate feedback, reference colors
    {"coef_led": 8, "coef_pol": 20},
    # Env 3: Moderate red shift + small feedback
    {
        "red": sample_truncnorm_integers(
            N, mean=100, std=25, low=0, high=255, random_state=SEED + 1
        ),
        "coef_led": 3,
        "coef_pol": 10,
    },
    # Env 4: Green shift + moderate feedback
    {
        "green": sample_truncnorm_integers(
            N, mean=80, std=25, low=0, high=255, random_state=SEED + 3
        ),
        "coef_led": 12,
        "coef_pol": 30,
    },
    # Env 5: Blue shift (lower) + large feedback
    {
        "blue": sample_truncnorm_integers(
            N, mean=60, std=20, low=0, high=255, random_state=SEED + 4
        ),
        "coef_led": 25,
        "coef_pol": 60,
    },
]

test_interventions = [
    # Test Env 0: Very strong feedback (extrapolates far beyond training range)
    #   → descendants will have a very different relationship with Y
    #   → Standard RF exploiting descendants will be punished
    {"coef_led": 40, "coef_pol": 90},
    # Test Env 1: Zero feedback + red shift (descendants uninformative about Y)
    #   → RF can't gain anything from descendants here
    {
        "red": sample_truncnorm_integers(
            N, mean=100, std=20, low=0, high=255, random_state=SEED_TEST + 1
        ),
    },
    # Test Env 2: Moderate feedback + green shift (mild extrapolation)
    {
        "coef_led": 6,
        "coef_pol": 15,
        "green": sample_truncnorm_integers(
            N, mean=100, std=20, low=0, high=255, random_state=SEED_TEST + 3
        ),
    },
    # Test Env 3: Zero feedback + blue shift (descendants uninformative about Y)
    {
        "blue": sample_truncnorm_integers(
            N, mean=55, std=20, low=0, high=255, random_state=SEED_TEST + 4
        ),
    },
    # Test Env 4: Extreme feedback + all colors shifted
    #   → maximum extrapolation stress test for the standard RF
    {
        "coef_led": 35,
        "coef_pol": 80,
        "red": sample_truncnorm_integers(
            N, mean=90, std=25, low=0, high=255, random_state=SEED_TEST + 5
        ),
        "green": sample_truncnorm_integers(
            N, mean=80, std=30, low=0, high=255, random_state=SEED_TEST + 6
        ),
        "blue": sample_truncnorm_integers(
            N, mean=60, std=20, low=0, high=255, random_state=SEED_TEST + 7
        ),
    },
]

all_interventions = {
    "train": train_interventions,
    "test": test_interventions,
}


dataset_name = "data/5_v1"


produce_dataset(dataset_type="train")
produce_dataset(dataset_type="test")
