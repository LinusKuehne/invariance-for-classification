import causalchamber.lab as lab
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

DIR = "/Users/linuskuehne/git/invariance-for-classification/scripts/causal_chambers/"

# initialize lab with credentials
rlab = lab.Lab(".credentials")


def sample_truncnorm_integers(n, mean, std, low, high, random_state=None):
    a = (low - mean) / std
    b = (high - mean) / std
    values = truncnorm.rvs(a, b, loc=mean, scale=std, size=n, random_state=random_state)
    return np.round(values).astype(int)


N = 200
SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------

# t_vis_3: [0,1,2,3], default 3
# diode_vis_3: [0,1], default 1

# t_ir_3: [0,1,2,3], default 3
# diode_ir_3: [0,1,2], default 2

# pol_1: [-270, 270], default 0
# pol_2: [-270, 270], default 0

state_0 = 0
state_1 = 25

interventions = []
env_indices = []

env_indices.append(0)
interventions.append({})

env_indices.append(1)
interventions.append({"t_ir_3": np.ones(N) * 1})

env_indices.append(2)
interventions.append(
    {
        "red": sample_truncnorm_integers(
            N, mean=10, std=20, low=0, high=255, random_state=SEED + 1
        ),
        "t_ir_3": np.ones(N) * 2,
    }
)

env_indices.append(3)
interventions.append(
    {
        "green": sample_truncnorm_integers(
            N, mean=100, std=20, low=0, high=255, random_state=SEED + 3
        )
    }
)

env_indices.append(4)
interventions.append(
    {
        "blue": sample_truncnorm_integers(
            N, mean=90, std=35, low=0, high=255, random_state=SEED + 4
        ),
        "diode_ir_3": np.ones(N) * 1,
    }
)

experiment_ids = []
for i, intervention in enumerate(interventions):
    inputs = {
        "pol_1": np.where(rng.integers(low=0, high=2, size=N) == 0, state_0, state_1),
        "red": sample_truncnorm_integers(
            N, mean=64, std=20, low=0, high=255, random_state=SEED + i + 101
        ),
        "green": sample_truncnorm_integers(
            N, mean=32, std=30, low=0, high=255, random_state=SEED + i + 102
        ),
        "blue": sample_truncnorm_integers(
            N, mean=90, std=12, low=0, high=255, random_state=SEED + i + 103
        ),
    }
    for target, values in intervention.items():
        inputs[target] = values

    experiment = rlab.new_experiment(chamber_id="lt-test-b0ni", config="standard")
    experiment.from_df(pd.DataFrame(inputs))
    experiment_id = experiment.submit(tag=f"env_{env_indices[i]}")
    experiment_ids.append(experiment_id)
    print(
        f"submitted experiment for environment {i + 1}: {env_indices[i]} ({experiment_id})"
    )

_ = rlab.get_experiments(print_max=len(env_indices) + 1)

dataframes = [
    rlab.download_data(eid, root=DIR + "tmp").dataframe for eid in experiment_ids
]

df = pd.concat(
    [df_tmp.assign(E=env_indices[i]) for i, df_tmp in enumerate(dataframes)],
    ignore_index=True,
)

df = df.assign(Y=np.where(df["pol_1"] < 0.00001, 0, 1))
df = df.drop(columns="pol_1")

df[["Y", "red", "green", "blue", "ir_3", "vis_3", "E"]].to_csv(
    DIR + "data/dataset_1_small.csv", index=False
)

df[
    [
        "Y",
        "red",
        "green",
        "blue",
        "ir_1",
        "vis_1",
        "ir_3",
        "vis_3",
        "E",
    ]
].to_csv(DIR + "data/dataset_1.csv", index=False)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

# import seaborn as sns

# env_order = sorted(df["E"].unique())
# palette_colors = sns.color_palette("colorblind", n_colors=len(env_order))
# palette_dict = {env_order[i]: palette_colors[i] for i in range(len(env_order))}

# sns.pairplot(
#     df,
#     x_vars=["red", "green", "blue"],
#     y_vars=["ir_1", "ir_2", "ir_3", "vis_1", "vis_2", "vis_3"],
#     palette=palette_dict,
#     hue="E",
#     markers=".",
#     plot_kws={"alpha": 1, "s": 75},
# )

# sns.pairplot(
#     df,
#     vars=["red", "green", "blue", "ir_1", "ir_3", "vis_1", "vis_3"],
#     hue="E",
#     palette=palette_dict,
# )

# # correlations
# df[["Y", "ir_3", "vis_3", "vis_1"]]
# df[["Y", "ir_3", "vis_3", "vis_1"]].corr()


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# df_test = df.loc[df.E == 0][
#     ["Y", "red", "green", "blue", "ir_1", "ir_2", "ir_3", "vis_1", "vis_2", "vis_3"]
# ]

# X_train = df_test.drop(columns=["Y"])
# y_train = df_test["Y"]

# rf = RandomForestClassifier(random_state=SEED, bootstrap=True, oob_score=True)
# rf.fit(X_train, y_train)
# print(rf.oob_score_)
