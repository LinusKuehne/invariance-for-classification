"""
Complex DGP for classification with interventions.

This module implements a more complex SCM with multiple
environments and interventions.

Causal structure:
    X1 -> X2
    X1, X2 -> Y
    Y -> X3 <- X5
    Y, X3, X7 -> X4
    Y, X4 -> X6
    X2 <- E -> X4
    Plus additional noise variables X8, X9, ...

    Stable blanket: {X1, X2, X3, X5}
    Invariant sets: HAVE to include {X1, X2}, and NOT {X4, X6}
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd

from invariance_for_classification.generate_data.synthetic_DGP import _nonlinear_f


def _runif_strong(
    n: int,
    min_val: float,
    max_val: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate uniformly distributed samples from [-max, -min] U [min, max].

    Parameters
    ----------
    n : int
        Number of samples
    min_val : float
        Non-negative minimum absolute value
    max_val : float
        Non-negative maximum absolute value (must be > min_val)
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        Array of length n with samples from the specified distribution
    """
    # Sample the absolute value
    abs_samples = rng.uniform(min_val, max_val, size=n)

    # Randomly sample the sign
    signs = rng.choice([-1, 1], size=n)

    return signs * abs_samples


def _sim_env_scm(
    n: int,
    d_1: float,
    d_2: float,
    env_name: str,
    num_noise: int,
    model: Literal["logistic_linear", "logistic_complex"],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate a sample from the SCM for one environment with given interventions.

    Parameters
    ----------
    n : int
        Sample size
    d_1 : float
        Intervention on X2
    d_2 : float
        Intervention on X4
    env_name : str
        Name of the environment
    num_noise : int
        Number of additional pure noise predictors (X8, X9, ...)
    model : {"logistic_linear", "logistic_complex"}
        Model type for Y. "logistic_linear" uses a simple linear link,
        "logistic_complex" uses a non-monotonic transformation with a peak,
        trough, and decay.
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    pd.DataFrame
        DataFrame with the generated sample
    """
    # Standard deviation of noise
    sd = 1.0

    # Scaling of the interventions
    a_1 = 1.5
    a_2 = 2.0

    # Generate X1 and X2
    X1 = rng.normal(0, sd, size=n)
    X2 = X1 + a_1 * d_1 + rng.normal(0, sd, size=n)

    # Y depends non-linearly on X1 and X2
    input_val = X1 + 0.75 * X2

    # Generate Y based on model type
    if model == "logistic_linear":
        Y = (rng.logistic(size=n) < 2 * input_val).astype(int)
    elif model == "logistic_complex":
        fx = _nonlinear_f(input_val)
        Y = (rng.logistic(size=n) < fx).astype(int)
    else:
        raise ValueError(
            f"Invalid model name: {model}. Choose 'logistic_linear' or 'logistic_complex'."
        )

    # Generate remaining variables
    X5 = rng.normal(0, sd, size=n)
    X3 = -Y + X5 + rng.normal(0, sd, size=n)

    X7 = rng.normal(0, sd, size=n)
    X4 = Y - 0.5 * X3 + X7 + a_2 * d_2 + rng.normal(0, sd / 2, size=n)

    X6 = Y - X4 + rng.normal(0, sd, size=n)

    # Build the data dictionary
    data = {
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "X4": X4,
        "X5": X5,
        "X6": X6,
        "X7": X7,
    }

    # Add pure noise predictors
    for i in range(num_noise):
        data[f"X{8 + i}"] = rng.normal(0, sd, size=n)

    # Add Y and environment
    data["Y"] = Y
    data["Env"] = np.full(n, env_name)

    return pd.DataFrame(data)


def generate_complex_scm_data(
    n_train: int,
    n_test: int,
    model: Literal["logistic_linear", "logistic_complex"] = "logistic_complex",
    int_strength_train: float = 1.0,
    int_strength_test: float = 1.5,
    num_noise: int = 6,
    n_envs: int = 10,
    seed: Optional[int] = 42,
) -> dict[str, pd.DataFrame]:
    """
    Generate training and testing samples from the complex SCM.

    Training environments have weaker interventions sampled uniformly from
    [-int_strength_train, int_strength_train].

    Testing environments have stronger interventions sampled uniformly from
    [-int_strength_test, -int_strength_train] U [int_strength_train, int_strength_test].

    Parameters
    ----------
    n_train : int
        Sample size for training (per environment)
    n_test : int
        Sample size for testing (per environment)
    model : {"logistic_linear", "logistic_complex"}, default="logistic_complex"
        Model type for Y. "logistic_linear" uses a simple linear link function,
        "logistic_complex" uses a non-monotonic transformation.
    int_strength_train : float, default=1.0
        Strength of interventions for training environments
    int_strength_test : float, default=1.5
        Strength of interventions for testing environments
    num_noise : int, default=6
        Number of additional pure noise predictors
    n_envs : int, default=10
        Number of environments for both training and testing
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys "sample_train" and "sample_test" containing
        the training and testing DataFrames respectively

    Examples
    --------
    >>> data = generate_complex_scm_data(
    ...     n_train=500,
    ...     n_test=500,
    ...     model="logistic_complex",
    ...     num_noise=3,
    ... )
    >>> sample_train = data["sample_train"]
    >>> sample_test = data["sample_test"]
    """
    rng = np.random.default_rng(seed=seed)

    # Sample intervention values for training environments
    d_1_train = rng.uniform(-int_strength_train, int_strength_train, size=n_envs)
    d_2_train = rng.uniform(-int_strength_train, int_strength_train, size=n_envs)

    # Generate training samples
    train_samples = []
    for i in range(n_envs):
        env_sample = _sim_env_scm(
            n=n_train,
            d_1=d_1_train[i],
            d_2=d_2_train[i],
            env_name=f"train{i + 1}",
            num_noise=num_noise,
            model=model,
            rng=rng,
        )
        train_samples.append(env_sample)

    sample_train = pd.concat(train_samples, ignore_index=True)

    # Sample intervention values for testing environments
    # Uniformly from [-int_strength_test, -int_strength_train] ∪ [int_strength_train, int_strength_test]
    d_1_test = _runif_strong(n_envs, int_strength_train, int_strength_test, rng)
    d_2_test = _runif_strong(n_envs, int_strength_train, int_strength_test, rng)

    # Generate testing samples
    test_samples = []
    for i in range(n_envs):
        env_sample = _sim_env_scm(
            n=n_test,
            d_1=d_1_test[i],
            d_2=d_2_test[i],
            env_name=f"test{i + 1}",
            num_noise=num_noise,
            model=model,
            rng=rng,
        )
        test_samples.append(env_sample)

    sample_test = pd.concat(test_samples, ignore_index=True)

    return {"sample_train": sample_train, "sample_test": sample_test}
