"""
Dataset 1
---------
We set pol_1 (source node) to discrete levels and Y is a deterministic function of pol_1 only (Y=1 iff pol_1 in pos_class_values).

1a: "Linear Y": pol_1 in {0, 30} with Y = 1{pol_1=30}.

1b: "Nonlinear Y": pol_1 in {0, 20, 30, 50} with Y = 1{pol_1 in {20, 30}}.
     vis_3 is monotonic in pol_1 (Malus' law), but Y is non-monotonic
     (Y=0: vis_3 at extremes, Y=1: vis_3 in mid-range)
     Linear classifiers on vis_3 cannot separate the classes;
     tree-based models can.

Stable blanket: {red, green, blue, vis_3}.
"""

import numpy as np
from utils import produce_dataset_1, sample_truncnorm_integers

N_TRAIN = 10000
N_TEST = 5000

POL_1_LEVELS_A = [0, 30]
POS_CLASS_VALUES_A = [30]
POL_1_LEVELS_B = [0, 20, 30, 50]
POS_CLASS_VALUES_B = [20, 30]

SEED = 42
SEED_TEST = SEED + 1000


train_interventions = [
    # Env 0: reference
    {},
    # Env 1: mild IR shift (default: 3)
    {"t_ir_3": np.ones(N_TRAIN) * 2},
    # Env 2: mild IR shift + colour shift
    {
        "t_ir_3": np.ones(N_TRAIN) * 2,
        "red": sample_truncnorm_integers(
            N_TRAIN, mean=130, std=20, low=0, high=255, random_state=SEED + 1
        ),
    },
    # Env 3: brighter red
    {
        "red": sample_truncnorm_integers(
            N_TRAIN, mean=140, std=20, low=0, high=255, random_state=SEED + 2
        ),
    },
    # Env 4: brighter green, darker blue
    {
        "green": sample_truncnorm_integers(
            N_TRAIN, mean=120, std=25, low=0, high=255, random_state=SEED + 3
        ),
        "blue": sample_truncnorm_integers(
            N_TRAIN, mean=70, std=20, low=0, high=255, random_state=SEED + 4
        ),
    },
    # Env 5: darker red, brighter blue
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
    # Test env 0: stronger IR shift
    {
        "t_ir_3": np.ones(N_TEST) * 1,
    },
    # Test env 1: IR shift + red within training range
    {
        "t_ir_3": np.ones(N_TEST) * 1,
        "red": sample_truncnorm_integers(
            N_TEST, mean=120, std=25, low=0, high=255, random_state=SEED_TEST + 1
        ),
    },
    # Test env 2: IR shift + green within training range
    {
        "t_ir_3": np.ones(N_TEST) * 1,
        "green": sample_truncnorm_integers(
            N_TEST, mean=100, std=20, low=0, high=255, random_state=SEED_TEST + 3
        ),
    },
    # Test env 3: IR shift + blue within training range
    {
        "t_ir_3": np.ones(N_TEST) * 1,
        "blue": sample_truncnorm_integers(
            N_TEST, mean=90, std=20, low=0, high=255, random_state=SEED_TEST + 4
        ),
    },
    # Test env 4: IR shift + all colours shifted
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


def main():
    # ---- dataset 1a -------------------------------------

    # generate training dataset
    produce_dataset_1(
        pos_class_values=POS_CLASS_VALUES_A,
        all_interventions=all_interventions,
        pol_1_levels=POL_1_LEVELS_A,
        dataset_type="train",
        seed=SEED,
        dataset_name="data/1a",
        n_per_env=N_TRAIN,
    )

    # generate test dataset
    produce_dataset_1(
        pos_class_values=POS_CLASS_VALUES_A,
        all_interventions=all_interventions,
        pol_1_levels=POL_1_LEVELS_A,
        dataset_type="test",
        seed=SEED_TEST,
        dataset_name="data/1a",
        n_per_env=N_TEST,
    )

    # ---- dataset 1b -------------------------------------

    # generate training dataset
    produce_dataset_1(
        pos_class_values=POS_CLASS_VALUES_B,
        all_interventions=all_interventions,
        pol_1_levels=POL_1_LEVELS_B,
        dataset_type="train",
        seed=SEED,
        dataset_name="data/1b",
        n_per_env=N_TRAIN,
    )

    # generate test dataset
    produce_dataset_1(
        pos_class_values=POS_CLASS_VALUES_B,
        all_interventions=all_interventions,
        pol_1_levels=POL_1_LEVELS_B,
        dataset_type="test",
        seed=SEED_TEST,
        dataset_name="data/1b",
        n_per_env=N_TEST,
    )


if __name__ == "__main__":
    main()
