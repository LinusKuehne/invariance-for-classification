"""
Complex data generating process (DGP) for classification with interventions.

This module implements a more complex SCM (Structural Causal Model) with multiple
environments and interventions.

Causal structure:
    X1 -> X2
    X1, X2 -> Y
    Y -> X3 <- X5
    Y, X3, X7 -> X4
    Y, X4 -> X6
    Plus additional noise variables X8, X9, ...

Interventions:
    d.1 affects X2 (scaled by a.1 = 1.5)
    d.2 affects X4 (scaled by a.2 = 2.0)
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd


def _nonlinear_fx(x: np.ndarray) -> np.ndarray:
    """
    Non-linear transformation function for the logistic_complex model.

    For x < 0: returns |x|
    For 0 <= x <= 3: polynomial
    For x > 3: monotonically decreasing (inverted logarithm shape)

    Parameters
    ----------
    x : np.ndarray
        Input values

    Returns
    -------
    np.ndarray
        Transformed values
    """
    result = np.zeros_like(x, dtype=float)

    # For x < 0: use |x|
    neg_mask = x < 0
    result[neg_mask] = np.abs(x[neg_mask])

    # For 0 <= x <= 3: polynomial
    # Using: f(x) = x * (x - 1.5) * (x - 2.5)
    mid_mask = (x >= 0) & (x <= 3)
    x_mid = x[mid_mask]
    result[mid_mask] = x_mid * (x_mid - 1.5) * (x_mid - 2.5)

    # For x > 3: monotonically decreasing like inverted logarithm
    # We need continuity at x = 3
    # f(3) = 3 * (3 - 1.5) * (3 - 2.5)
    # Use: f(x) = f(3) - ln(1 + k * (x - 3)) for x > 3
    high_mask = x > 3
    x_high = x[high_mask]
    f_at_transition = 3 * (3 - 1.5) * (3 - 2.5)  # value at x = 3
    k = 4.0  # controls how fast it decreases
    result[high_mask] = f_at_transition - np.log(1 + k * (x_high - 3))

    return result


# def plot_nonlinear_fx(
#     x_min: float = -3.0,
#     x_max: float = 6.0,
#     n_points: int = 500,
#     save_path: Optional[str] = None,
# ) -> None:
#     """
#     Plot the non-linear transformation function used in the complex DGP.

#     Parameters
#     ----------
#     x_min : float
#         Minimum x value for the plot
#     x_max : float
#         Maximum x value for the plot
#     n_points : int
#         Number of points to plot
#     save_path : str, optional
#         If provided, save the figure to this path
#     """
#     import matplotlib.pyplot as plt

#     x = np.linspace(x_min, x_max, n_points)
#     y = _nonlinear_fx(x)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(x, y, "b-", linewidth=2, label=r"$f(x)$")
#     ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
#     ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
#     ax.axvline(x=3.2, color="red", linestyle=":", alpha=0.5, label="Transition (x=3.2)")

#     # Mark key features
#     ax.scatter([0], [0], color="red", s=50, zorder=5, label="Start (0, 0)")

#     # Find approximate peak and trough for 0 < x <= 3.2
#     x_mid = np.linspace(0.01, 3.2, 500)
#     y_mid = _nonlinear_fx(x_mid)
#     peak_idx = np.argmax(y_mid)
#     trough_idx = np.argmin(y_mid)

#     ax.scatter(
#         [x_mid[peak_idx]],
#         [y_mid[peak_idx]],
#         color="green",
#         s=50,
#         zorder=5,
#         label=f"Peak (~{x_mid[peak_idx]:.2f})",
#     )
#     ax.scatter(
#         [x_mid[trough_idx]],
#         [y_mid[trough_idx]],
#         color="orange",
#         s=50,
#         zorder=5,
#         label=f"Trough (~{x_mid[trough_idx]:.2f})",
#     )

#     ax.set_xlabel("x (input)", fontsize=12)
#     ax.set_ylabel("f(x)", fontsize=12)
#     ax.set_title("Non-linear transformation function for 'logistic_complex' model", fontsize=14)
#     ax.legend(loc="upper right")
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"Figure saved to {save_path}")

#     plt.show()


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
        fx = _nonlinear_fx(input_val)
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


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     # Example usage and plot the non-linear function
#     print("Plotting non-linear transformation function...")
#     plot_nonlinear_fx()

#     print("\nAnalyzing input support (range of input = X1 + 0.75 * X2)...")
#     # Generate data to check input range
#     data = generate_complex_scm_data(
#         n_train=500,
#         n_test=500,
#         model="logistic_linear",  # Use logistic_linear to see input before transformation
#         num_noise=3,
#         seed=42,
#     )

#     # Reconstruct input values
#     train_df = data["sample_train"]
#     test_df = data["sample_test"]

#     input_train = train_df["X1"] + 0.75 * train_df["X2"]
#     input_test = test_df["X1"] + 0.75 * test_df["X2"]

#     print(f"\nTraining input statistics:")
#     print(f"  Min: {input_train.min():.3f}")
#     print(f"  Max: {input_train.max():.3f}")
#     print(f"  Mean: {input_train.mean():.3f}")
#     print(f"  Std: {input_train.std():.3f}")
#     print(f"  5th percentile: {input_train.quantile(0.05):.3f}")
#     print(f"  95th percentile: {input_train.quantile(0.95):.3f}")

#     print(f"\nTesting input statistics (stronger interventions):")
#     print(f"  Min: {input_test.min():.3f}")
#     print(f"  Max: {input_test.max():.3f}")
#     print(f"  Mean: {input_test.mean():.3f}")
#     print(f"  Std: {input_test.std():.3f}")
#     print(f"  5th percentile: {input_test.quantile(0.05):.3f}")
#     print(f"  95th percentile: {input_test.quantile(0.95):.3f}")

#     # Plot histogram of input values
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#     axes[0].hist(input_train, bins=50, edgecolor="black", alpha=0.7)
#     axes[0].set_xlabel("Input (X1 + 0.75*X2)")
#     axes[0].set_ylabel("Frequency")
#     axes[0].set_title("Training: Input Distribution")
#     axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.7)
#     axes[0].axvline(x=3.2, color="orange", linestyle="--", alpha=0.7, label="x=3.2")

#     axes[1].hist(input_test, bins=50, edgecolor="black", alpha=0.7, color="green")
#     axes[1].set_xlabel("Input (X1 + 0.75*X2)")
#     axes[1].set_ylabel("Frequency")
#     axes[1].set_title("Testing: Input Distribution")
#     axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.7)
#     axes[1].axvline(x=3.2, color="orange", linestyle="--", alpha=0.7, label="x=3.2")

#     plt.tight_layout()
#     plt.show()

#     print("\n" + "=" * 60)
#     print("Generating example data with 'logistic_complex' model...")
#     data = generate_complex_scm_data(
#         n_train=500,
#         n_test=500,
#         model="logistic_complex",
#         num_noise=3,
#     )

#     print(f"\nTraining data shape: {data['sample_train'].shape}")
#     print(f"Testing data shape: {data['sample_test'].shape}")
#     print(
#         f"\nClass balance in training: {data['sample_train']['Y'].value_counts().to_dict()}"
#     )
