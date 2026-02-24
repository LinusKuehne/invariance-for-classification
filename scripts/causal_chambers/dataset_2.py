"""
Dataset 2
---------
Two-stage data generation process with feedback loop.
1. Generate R/G/B -> measure ir_1 -> compute Y
2. Set led_3_ir, pol_2 based on Y -> measure everything

Stable blanket: {red, green, blue}.
"""

import os
import time

import causalchamber.lab as lab
import numpy as np
import pandas as pd
from utils import SAVE_COLS, sample_truncnorm_integers, wait_for_completion

DIR = os.path.dirname(os.path.abspath(__file__))

N_TRAIN = 10000
N_TEST = 5000

# threshold on ir_1 to determine Y=0 vs Y=1
THRESHOLD = 12500

SEED = 42
SEED_TEST = SEED + 1000

rlab = lab.Lab(os.path.join(DIR, ".credentials"))


def produce_dataset_2(dataset_type="train"):
    if dataset_type == "train":
        seed = SEED
        n = N_TRAIN
    else:
        seed = SEED_TEST
        n = N_TEST

    interventions = []
    env_indices = []

    # unpack correct interventions <-> environments
    for i, intervention_dict in enumerate(all_interventions[dataset_type]):
        env_indices.append(i)
        interventions.append(intervention_dict.copy())

    # --- phase 1: prior experiments ---
    experiment_ids_phase1 = []
    phase1_inputs_list = []

    print(f"Submitting phase 1 experiments ({dataset_type})...")
    for i, intervention in enumerate(interventions):
        # start with reference setting and then apply intervention (if applicable)
        inputs = reference_setting_2(seed + i, n)

        # apply interventions if applicable (excluding feedback params which are used in phase 2)
        feedback_keys = {"coef_led", "coef_pol", "base_led", "base_pol"}
        for target, values in intervention.items():
            if target not in feedback_keys:
                inputs[target] = values

        phase1_inputs_list.append(inputs)

        # submit experiment
        experiment = rlab.new_experiment(chamber_id="lt-test-b0ni", config="standard")
        experiment.from_df(pd.DataFrame(inputs))
        experiment_id = experiment.submit(
            tag=f"{dataset_type}_env_{env_indices[i]}_prior"
        )
        experiment_ids_phase1.append(experiment_id)

        # avoid submitting experiments too quickly (not to trigger firewall)
        time.sleep(2)

    print(f"Waiting for phase 1 ({dataset_type})...")
    wait_for_completion(rlab)

    # --- phase 2: feedback & post experiments ---
    experiment_ids_phase2 = []
    Ys_list = []

    print(f"Submitting phase 2 experiments ({dataset_type})...")
    for i, prior_eid in enumerate(experiment_ids_phase1):
        # download phase 1 results
        try:
            df_prior = rlab.download_data(prior_eid, root=DIR + "/tmp").dataframe
        except Exception as e:
            print(f"Error downloading experiment {prior_eid}: {e}")
            raise

        ir_1_prior = df_prior["ir_1"].to_numpy()

        # compute Y
        Y = np.where(ir_1_prior > THRESHOLD, 1, 0)
        Ys_list.append(Y)

        # determine coefficients (default to 0 if not specified)
        intervention = interventions[i]
        base_led = intervention.get("base_led", 0)
        base_pol = intervention.get("base_pol", 0)
        coef_led = intervention.get("coef_led", 0)
        coef_pol = intervention.get("coef_pol", 0)

        # prepare phase 2 inputs
        inputs_phase2 = phase1_inputs_list[i].copy()

        # add feedback interventions
        # led_3_ir = base_led + Y * coef_led
        # pol_2   = base_pol + Y * coef_pol
        # with base > 0 and coef < 0, the descendant–Y relationship reverses.
        # LED parameters accept non-negative integers only.
        inputs_phase2["led_3_ir"] = np.clip(base_led + Y * coef_led, 0, None).astype(
            int
        )
        inputs_phase2["pol_2"] = np.clip(base_pol + Y * coef_pol, 0, None).astype(int)

        # submit phase 2 experiment
        experiment = rlab.new_experiment(chamber_id="lt-test-b0ni", config="standard")
        experiment.from_df(pd.DataFrame(inputs_phase2))
        experiment_id = experiment.submit(
            tag=f"{dataset_type}_env_{env_indices[i]}_post"
        )
        experiment_ids_phase2.append(experiment_id)

    print(f"Waiting for phase 2 ({dataset_type})...")
    wait_for_completion(rlab)

    # --- collect data ---
    print(f"Downloading phase 2 data ({dataset_type})...")
    dataframes = []
    for i, eid in enumerate(experiment_ids_phase2):
        df_post = rlab.download_data(eid, root=DIR + "/tmp").dataframe

        # attach Y and environment index
        df_post = df_post.assign(Y=Ys_list[i])
        df_post = df_post.assign(E=env_indices[i])

        dataframes.append(df_post)

    df = pd.concat(dataframes, ignore_index=True)

    df[SAVE_COLS].to_csv(DIR + f"/data/2_{dataset_type}.csv", index=False)


def reference_setting_2(random_state, n):
    """Generate reference setting for dataset 2 (distr. of variables not specifically intervened on)."""
    inputs = {
        "red": sample_truncnorm_integers(
            n,
            mean=64,
            std=20,
            low=0,
            high=255,
            random_state=random_state + 11,
        ),
        "green": sample_truncnorm_integers(
            n,
            mean=32,
            std=30,
            low=0,
            high=255,
            random_state=random_state + 12,
        ),
        "blue": sample_truncnorm_integers(
            n,
            mean=90,
            std=12,
            low=0,
            high=255,
            random_state=random_state + 13,
        ),
    }

    return inputs


train_interventions = [
    # Env 0: reference
    {},
    # Env 1: strong feedback, reference colors
    {"coef_led": 20, "coef_pol": 50},
    # Env 2: moderate feedback, reference colors
    {"coef_led": 8, "coef_pol": 20},
    # Env 3: moderate red shift + small feedback
    {
        "red": sample_truncnorm_integers(
            N_TRAIN, mean=100, std=25, low=0, high=255, random_state=SEED + 1
        ),
        "coef_led": 3,
        "coef_pol": 10,
    },
    # Env 4: green shift + moderate feedback
    {
        "green": sample_truncnorm_integers(
            N_TRAIN, mean=80, std=25, low=0, high=255, random_state=SEED + 3
        ),
        "coef_led": 12,
        "coef_pol": 30,
    },
    # Env 5: blue shift (lower) + large feedback
    {
        "blue": sample_truncnorm_integers(
            N_TRAIN, mean=60, std=20, low=0, high=255, random_state=SEED + 4
        ),
        "coef_led": 25,
        "coef_pol": 60,
    },
]

test_interventions = [
    # Test env 0: reversed feedback (base > 0, coef < 0)
    {"base_led": 25, "coef_led": -25, "base_pol": 60, "coef_pol": -60},
    # Test env 1: no feedback + red shift
    {
        "red": sample_truncnorm_integers(
            N_TEST, mean=100, std=20, low=0, high=255, random_state=SEED_TEST + 1
        ),
    },
    # Test env 2: moderate feedback + green shift
    {
        "coef_led": 6,
        "coef_pol": 15,
        "green": sample_truncnorm_integers(
            N_TEST, mean=100, std=20, low=0, high=255, random_state=SEED_TEST + 3
        ),
    },
    # Test env 3: reversed feedback + blue shift
    {
        "base_led": 20,
        "coef_led": -20,
        "base_pol": 50,
        "coef_pol": -50,
        "blue": sample_truncnorm_integers(
            N_TEST, mean=55, std=20, low=0, high=255, random_state=SEED_TEST + 4
        ),
    },
    # Test env 4: extreme positive feedback + all colors shifted
    {
        "coef_led": 35,
        "coef_pol": 80,
        "red": sample_truncnorm_integers(
            N_TEST, mean=90, std=25, low=0, high=255, random_state=SEED_TEST + 5
        ),
        "green": sample_truncnorm_integers(
            N_TEST, mean=80, std=30, low=0, high=255, random_state=SEED_TEST + 6
        ),
        "blue": sample_truncnorm_integers(
            N_TEST, mean=60, std=20, low=0, high=255, random_state=SEED_TEST + 7
        ),
    },
]

all_interventions = {
    "train": train_interventions,
    "test": test_interventions,
}


def main():
    produce_dataset_2(dataset_type="train")
    produce_dataset_2(dataset_type="test")


if __name__ == "__main__":
    main()
