"""
Causal chambers parameters
--------------------------
- t_ir_3  : [0,1,2,3], default 3   (integration time of IR sensor 3)
- diode_ir_3 : [0,1,2], default 2  (IR diode setting)
- pol_1   : [-270, 270], default 0  (polariser angle → determines Y)
- red/green/blue : [0, 255]         (LED colour inputs)

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
    else:
        seed = SEED_TEST

    rng = np.random.default_rng(seed)

    interventions = []
    env_indices = []

    for i, intervention_dict in enumerate(all_interventions[dataset_type]):
        env_indices.append(i)
        interventions.append(intervention_dict)

    experiment_ids = []
    for i, intervention in enumerate(interventions):
        inputs = reference_setting(seed + i, rng)

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

    df = df.assign(Y=np.where(df["pol_1"] < 0.00001, 0, 1))
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


N = 200
SEED = 42
SEED_TEST = SEED + 1000

state_0 = 0
state_1 = 30


def reference_setting(random_state, rng):
    inputs = {
        "pol_1": np.sort(
            np.where(rng.integers(low=0, high=2, size=N) == 0, state_0, state_1)
        ),
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


train_interventions = [
    {},
    {"t_ir_3": np.ones(N) * 2},
    {
        "red": sample_truncnorm_integers(
            N, mean=90, std=30, low=0, high=255, random_state=SEED + 1
        ),
    },
    {
        "green": sample_truncnorm_integers(
            N, mean=64, std=25, low=0, high=255, random_state=SEED + 3
        ),
    },
    {
        "blue": sample_truncnorm_integers(
            N, mean=40, std=20, low=0, high=255, random_state=SEED + 4
        ),
    },
]

test_interventions = [
    {"t_ir_3": np.ones(N) * 1, "diode_ir_3": np.ones(N) * 0},
    {
        "t_ir_3": np.ones(N) * 0,
        "red": sample_truncnorm_integers(
            N, mean=120, std=20, low=0, high=255, random_state=SEED_TEST + 1
        ),
    },
    {
        "t_ir_3": np.ones(N) * 1,
        "green": sample_truncnorm_integers(
            N, mean=100, std=20, low=0, high=255, random_state=SEED_TEST + 3
        ),
    },
    {
        "t_ir_3": np.ones(N) * 0,
        "blue": sample_truncnorm_integers(
            N, mean=50, std=20, low=0, high=255, random_state=SEED_TEST + 4
        ),
    },
    {
        "t_ir_3": np.ones(N) * 1,
        "diode_ir_3": np.ones(N) * 1,
        "red": sample_truncnorm_integers(
            N, mean=90, std=25, low=0, high=255, random_state=SEED_TEST + 5
        ),
        "green": sample_truncnorm_integers(
            N, mean=80, std=30, low=0, high=255, random_state=SEED_TEST + 6
        ),
        "blue": sample_truncnorm_integers(
            N, mean=100, std=20, low=0, high=255, random_state=SEED_TEST + 7
        ),
    },
]

all_interventions = {
    "train": train_interventions,
    "test": test_interventions,
}


dataset_name = "data/2_v2"


produce_dataset(dataset_type="train")
produce_dataset(dataset_type="test")
