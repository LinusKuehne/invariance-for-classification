"""
1b: Nonlinear Y. pol_1 ∈ {0, 20, 30, 50} with Y = 1{pol_1 ∈ {20, 30}}.
     vis_3 is monotonic in pol_1 (Malus' law), but Y is non-monotonic.
     Y=0 → vis_3 at extremes (high@0°, low@50°),
     Y=1 → vis_3 in mid-range (cos²(20°)≈0.88, cos²(40°)≈0.59).
     Linear classifiers on vis_3 cannot separate the classes;
     tree-based models can.
Stable blanket: {red, green, blue, vis_3}.
"""

import time

import causalchamber.lab as lab
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

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
        n = N_TRAIN
    else:
        seed = SEED_TEST
        n = N_TEST

    rng = np.random.default_rng(seed)

    interventions = []
    env_indices = []

    for i, intervention_dict in enumerate(all_interventions[dataset_type]):
        env_indices.append(i)
        interventions.append(intervention_dict)

    experiment_ids = []
    for i, intervention in enumerate(interventions):
        inputs = reference_setting(seed + i, rng, n)

        for target, values in intervention.items():
            inputs[target] = values

        experiment = rlab.new_experiment(chamber_id="lt-test-b0ni", config="standard")
        experiment.from_df(pd.DataFrame(inputs))
        experiment_id = experiment.submit(tag=f"{dataset_type}_env_{env_indices[i]}")
        experiment_ids.append(experiment_id)

    status = rlab.get_experiments(verbose=False)[0]["status"]
    while not status == "DONE":
        time.sleep(15)
        status = rlab.get_experiments(verbose=False)[0]["status"]

    # collect dataframes and shuffle rows within each environment
    dataframes = [
        rlab.download_data(eid, root=DIR + "tmp")
        .dataframe.sample(frac=1, random_state=seed)
        .reset_index(drop=True)
        for eid in experiment_ids
    ]

    df = pd.concat(
        [df_tmp.assign(E=env_indices[i]) for i, df_tmp in enumerate(dataframes)],
        ignore_index=True,
    )

    # Nonlinear Y: Y=1 iff pol_1 ∈ {20, 30}, Y=0 iff pol_1 ∈ {0, 50}
    df = df.assign(Y=np.where(df["pol_1"].isin([20, 30]), 1, 0))
    df = df.drop(columns="pol_1")

    df[["Y", "red", "green", "blue", "ir_3", "vis_3", "E"]].to_csv(
        DIR + f"{dataset_name}" + f"_small_{dataset_type}.csv", index=False
    )

    df[["Y", "red", "green", "blue", "ir_1", "vis_1", "ir_3", "vis_3", "E"]].to_csv(
        DIR + f"{dataset_name}" + f"_{dataset_type}.csv", index=False
    )

    from sklearn.ensemble import RandomForestClassifier

    df_test = df.loc[df.E == 0][
        ["Y", "red", "green", "blue", "ir_1", "ir_2", "ir_3", "vis_1", "vis_2", "vis_3"]
    ]

    X_train = df_test.drop(columns=["Y"])
    y_train = df_test["Y"]

    rf = RandomForestClassifier(random_state=SEED, bootstrap=True, oob_score=True)
    rf.fit(X_train, y_train)
    print(rf.oob_score_)


N_TRAIN = 1000
N_TEST = 1000
SEED = 42
SEED_TEST = SEED + 1000

POL_1_LEVELS = [0, 20, 30, 50]


def reference_setting(random_state, rng, n):
    inputs = {
        "pol_1": np.sort(rng.choice(POL_1_LEVELS, size=n)),
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


train_interventions = [
    # Env 0 – reference
    {},
    # Env 1 – mild IR shift
    {"t_ir_3": np.ones(N_TRAIN) * 2},
    # Env 2 – mild IR shift + colour shift
    {
        "t_ir_3": np.ones(N_TRAIN) * 2,
        "red": sample_truncnorm_integers(
            N_TRAIN, mean=130, std=20, low=0, high=255, random_state=SEED + 1
        ),
    },
    # Env 3 – brighter red
    {
        "red": sample_truncnorm_integers(
            N_TRAIN, mean=140, std=20, low=0, high=255, random_state=SEED + 2
        ),
    },
    # Env 4 – brighter green, dimmer blue
    {
        "green": sample_truncnorm_integers(
            N_TRAIN, mean=120, std=25, low=0, high=255, random_state=SEED + 3
        ),
        "blue": sample_truncnorm_integers(
            N_TRAIN, mean=70, std=20, low=0, high=255, random_state=SEED + 4
        ),
    },
    # Env 5 – dimmer red, brighter blue
    {
        "red": sample_truncnorm_integers(
            N_TRAIN, mean=60, std=20, low=0, high=255, random_state=SEED + 5
        ),
        "blue": sample_truncnorm_integers(
            N_TRAIN, mean=150, std=20, low=0, high=255, random_state=SEED + 6
        ),
    },
]

test_interventions = [
    # Env 0 – IR shift only
    {
        "t_ir_3": np.ones(N_TEST) * 1,
    },
    # Env 1 – IR shift + red
    {
        "t_ir_3": np.ones(N_TEST) * 1,
        "red": sample_truncnorm_integers(
            N_TEST, mean=120, std=25, low=0, high=255, random_state=SEED_TEST + 1
        ),
    },
    # Env 2 – IR shift + green
    {
        "t_ir_3": np.ones(N_TEST) * 1,
        "green": sample_truncnorm_integers(
            N_TEST, mean=100, std=20, low=0, high=255, random_state=SEED_TEST + 3
        ),
    },
    # Env 3 – IR shift + blue
    {
        "t_ir_3": np.ones(N_TEST) * 1,
        "blue": sample_truncnorm_integers(
            N_TEST, mean=90, std=20, low=0, high=255, random_state=SEED_TEST + 4
        ),
    },
    # Env 4 – IR shift + all colours shifted
    {
        "t_ir_3": np.ones(N_TEST) * 1,
        "red": sample_truncnorm_integers(
            N_TEST, mean=110, std=25, low=0, high=255, random_state=SEED_TEST + 5
        ),
        "green": sample_truncnorm_integers(
            N_TEST, mean=90, std=20, low=0, high=255, random_state=SEED_TEST + 6
        ),
        "blue": sample_truncnorm_integers(
            N_TEST, mean=100, std=20, low=0, high=255, random_state=SEED_TEST + 7
        ),
    },
]

all_interventions = {
    "train": train_interventions,
    "test": test_interventions,
}


dataset_name = "data/1b"


produce_dataset(dataset_type="train")
produce_dataset(dataset_type="test")
