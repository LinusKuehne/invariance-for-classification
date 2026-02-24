"""Utility functions for generating causal chambers datasets."""

import os
import time

import causalchamber.lab as lab
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

SAVE_COLS = [
    "Y",
    "red",
    "green",
    "blue",
    "ir_1",
    "vis_1",
    "ir_2",
    "vis_2",
    "ir_3",
    "vis_3",
    "led_3_ir",
    "pol_2",
    "E",
]


def sample_truncnorm_integers(n, mean, std, low, high, random_state=None):
    """Sample integers from a truncated normal distribution."""
    a = (low - mean) / std
    b = (high - mean) / std
    values = truncnorm.rvs(a, b, loc=mean, scale=std, size=n, random_state=random_state)
    return np.round(values).astype(int)


def wait_for_completion(rlab):
    """Check experiment status every 15 seconds and wait until done."""
    status = rlab.get_experiments(verbose=False)[0]["status"]
    while not status == "DONE":
        time.sleep(15)
        status = rlab.get_experiments(verbose=False)[0]["status"]


def reference_setting_1(random_state, rng, n, pol_1_levels):
    """Generate reference setting for dataset 1 (distr. of variables not specifically intervened on)."""
    inputs = {
        "pol_1": np.sort(rng.choice(pol_1_levels, size=n)),
        "red": sample_truncnorm_integers(
            n,
            mean=100,
            std=25,
            low=0,
            high=255,
            random_state=random_state + 11,
        ),
        "green": sample_truncnorm_integers(
            n,
            mean=80,
            std=20,
            low=0,
            high=255,
            random_state=random_state + 12,
        ),
        "blue": sample_truncnorm_integers(
            n,
            mean=110,
            std=15,
            low=0,
            high=255,
            random_state=random_state + 13,
        ),
    }

    return inputs


def produce_dataset_1(
    pos_class_values,
    all_interventions,
    pol_1_levels,
    dataset_type="train",
    seed=42,
    dataset_name="data/1a",
    n_per_env=1000,
):
    """Generate causal chambers data and store it as csv."""
    rlab = lab.Lab(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".credentials")
    )
    rng = np.random.default_rng(seed)

    interventions = []
    env_indices = []

    # unpack correct interventions <-> environments
    for i, intervention_dict in enumerate(all_interventions[dataset_type]):
        env_indices.append(i)
        interventions.append(intervention_dict)

    experiment_ids = []
    print(f"Submitting {dataset_type} experiments...")
    for i, intervention in enumerate(interventions):
        # start with reference setting and then apply intervention (if applicable)
        inputs = reference_setting_1(seed + i, rng, n_per_env, pol_1_levels)
        for target, values in intervention.items():
            inputs[target] = values

        # submit experiment
        experiment = rlab.new_experiment(chamber_id="lt-test-b0ni", config="standard")
        experiment.from_df(pd.DataFrame(inputs))
        experiment_id = experiment.submit(tag=f"{dataset_type}_env_{env_indices[i]}")
        experiment_ids.append(experiment_id)

        # avoid submitting experiments too quickly (not to trigger firewall)
        time.sleep(2)

    print(f"Waiting for {dataset_type} experiments to complete...")
    wait_for_completion(rlab)

    # collect dataframes and shuffle rows within each environment (we generate data ordered by pol_1, so we need fewer polarizer motor movements)
    print(f"Downloading {dataset_type} data...")
    dataframes = [
        rlab.download_data(eid, root=DIR + "tmp")
        .dataframe.sample(frac=1, random_state=seed)
        .reset_index(drop=True)
        for eid in experiment_ids
    ]

    # concatenate dataframes and assign environment labels
    df = pd.concat(
        [df_tmp.assign(E=env_indices[i]) for i, df_tmp in enumerate(dataframes)],
        ignore_index=True,
    )

    # assign class labels based on pol_1 values and drop pol_1
    df = df.assign(Y=np.where(df["pol_1"].isin(pos_class_values), 1, 0))
    df = df.drop(columns="pol_1")

    df[SAVE_COLS].to_csv(DIR + f"{dataset_name}_{dataset_type}.csv", index=False)
