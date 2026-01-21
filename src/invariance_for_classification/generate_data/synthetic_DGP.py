from typing import Literal, Optional, overload

import numpy as np
import pandas as pd


def _nonlinear_f(x: np.ndarray) -> np.ndarray:
    """
    Non-linear transformation function for the logistic_complex model.

    For x < 0: returns |x|
    For 0 <= x <= 3: polynomial with peak and trough
    For x > 3: monotonically decreasing (inverted logarithm shape)

    This creates a non-monotonic relationship between X1 and Y, making the
    classification problem more challenging for linear models.
    """
    result = np.zeros_like(x, dtype=float)

    # For x < 0: use |x|
    neg_mask = x < 0
    result[neg_mask] = np.abs(x[neg_mask])

    # For 0 <= x <= 3: polynomial f(x) = x * (x - 1.5) * (x - 2.5)
    mid_mask = (x >= 0) & (x <= 3)
    x_mid = x[mid_mask]
    result[mid_mask] = x_mid * (x_mid - 1.5) * (x_mid - 2.5)

    # For x > 3: monotonically decreasing
    high_mask = x > 3
    x_high = x[high_mask]
    f_at_transition = 3 * (3 - 1.5) * (3 - 2.5)
    k = 4.0
    result[high_mask] = f_at_transition - np.log(1 + k * (x_high - 3))

    return result


def _simple_scm_one_env(
    n: int,
    int_val: float,
    env_idx: int,
    rng: np.random.Generator,
    model: Literal["logistic_linear", "logistic_complex"] = "logistic_linear",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    generates a sample from the simple SCM for a single environment.

    Causal graph:
    E -> X1 -> Y -> X2 <- E
    Y -> X3

    Parameters
    ----------
    n : int
        number of observations
    int_val : float
        intervention value in this environment (used in the SCM)
    env_idx : int
        index of the environment
    rng : np.random.Generator
        random number generator
    model : {"logistic_linear", "logistic_complex"}, default="logistic_linear"
        Model type for generating Y.
        - "logistic_linear": Y depends linearly on X1 via logistic link
        - "logistic_complex": Y depends non-monotonically on X1 via a
          polynomial transformation

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Tuple (df, int_df) where df contains the generated data with column
        ``E`` as the environment index, and int_df contains a single column
        ``int_value`` with the intervention value used to generate each row.
    """
    # structural parameters
    alpha = -1.0
    beta = 1.0
    gamma = 1.5

    # E is the environment index; int_value is the intervention strength used in the SCM
    E = np.full(n, env_idx)
    int_vals = np.full(n, int_val)

    X1 = alpha * int_vals + rng.normal(size=n)

    noise = rng.logistic(size=n)
    input_val = beta * X1

    # Generate Y based on model type
    if model == "logistic_linear":
        Y = (noise < input_val).astype(int)
    elif model == "logistic_complex":
        fx = _nonlinear_f(input_val)
        Y = (noise < fx).astype(int)
    else:
        raise ValueError(
            f"Invalid model: {model}. Choose 'logistic_linear' or 'logistic_complex'."
        )

    X2 = Y + gamma * int_vals + rng.normal(size=n)
    X3 = Y + rng.normal(size=n)

    df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "Y": Y, "E": E})
    int_df = pd.DataFrame({"int_value": int_vals})
    return df, int_df


@overload
def generate_scm_data(
    n_per_env: int,
    int_vals: Optional[list[float]] = None,
    seed: Optional[int] = 42,
    *,
    model: Literal["logistic_linear", "logistic_complex"] = "logistic_linear",
    return_int_values: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def generate_scm_data(
    n_per_env: int,
    int_vals: Optional[list[float]] = None,
    seed: Optional[int] = 42,
    *,
    model: Literal["logistic_linear", "logistic_complex"] = "logistic_linear",
    return_int_values: Literal[True],
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def generate_scm_data(
    n_per_env: int,
    int_vals: Optional[list[float]] = None,
    seed: Optional[int] = 42,
    *,
    model: Literal["logistic_linear", "logistic_complex"] = "logistic_linear",
    return_int_values: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    generates data from the simple SCM across multiple environments.

    Parameters
    ----------
    n_per_env : int
        number of observations per environment
    int_vals : list[float], optional
        list of intervention values (one per environment)
    seed : int, optional
        random seed for reproducibility
    model : {"logistic_linear", "logistic_complex"}, default="logistic_linear"
        Model type for generating Y.
        - "logistic_linear": Y depends linearly on X1 via logistic link
        - "logistic_complex": Y depends non-monotonically on X1 via a
          polynomial transformation
    return_int_values : bool, default=False
        If True, also return a second DataFrame with a single column
        ``int_value`` holding the intervention values used to generate each row.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, pd.DataFrame)
        If ``return_int_values=False`` (default), returns df containing the
        generated data across all environments with ``E`` as the environment
        index.

        If ``return_int_values=True``, returns (df, int_df) where int_df
        contains a single column ``int_value`` holding the intervention values
        used to generate each row.
    """
    if int_vals is None:
        int_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]

    rng = np.random.default_rng(seed=seed)
    all_data = []
    all_int = []
    for env_idx, int_val in enumerate(int_vals):
        env_data, env_int = _simple_scm_one_env(
            n_per_env, int_val, env_idx, rng, model=model
        )
        all_data.append(env_data)
        all_int.append(env_int)

    df = pd.concat(all_data, ignore_index=True)
    if not return_int_values:
        return df

    int_df = pd.concat(all_int, ignore_index=True)
    return df, int_df
