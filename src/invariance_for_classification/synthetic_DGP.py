from typing import Optional

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)


def _sample_disjoint_uniform(n: int, low: float, high: float) -> np.ndarray:
    """
    Sample n observations uniformly from the union [-high, -low] U [low, high].
    """
    magnitudes = rng.uniform(low, high, size=n)
    signs = rng.choice([-1, 1], size=n, replace=True)
    return signs * magnitudes


def _simple_scm_one_env(
    n: int,
    env_val: float,
    env_idx: int,
) -> pd.DataFrame:
    """Generates a sample from the simple SCM for a single environment.

    Causal graph:
    E -> X1 -> Y -> X2 <- E
    Y -> X3

    Args:
        n (int): number of observations
        env_val (float): value of variable E in this environment
        env_name (str): name of the environment

    Returns:
        pd.DataFrame: DataFrame containing the generated data
    """
    # structural parameters
    alpha = -1.0
    beta = 1.0
    gamma = 1.5

    # intervention vector
    E = np.full(n, env_val)

    X1 = alpha * E + rng.normal(size=n)

    noise = rng.logistic(size=n)
    input_val = beta * X1
    Y = (noise < input_val).astype(int)

    X2 = Y + gamma * E + rng.normal(size=n)
    X3 = Y + rng.normal(size=n)

    # construct DataFrame
    data = {"X1": X1, "X2": X2, "X3": X3, "Y": Y, "E": E, "Env": np.full(n, env_idx)}

    return pd.DataFrame(data)


def generate_scm_data(
    n_per_env: int,
    env_values: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Generates data from the simple SCM across multiple environments.

    Args:
        n_per_env (int): number of observations per environment
        env_values (list[float]): list of environment values for E in each environment

    Returns:
        pd.DataFrame: DataFrame containing the generated data across all environments
    """
    if env_values is None:
        env_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    all_data = []
    for env_idx, env_val in enumerate(env_values):
        env_data = _simple_scm_one_env(n_per_env, env_val, env_idx)
        all_data.append(env_data)
    return pd.concat(all_data, ignore_index=True)
