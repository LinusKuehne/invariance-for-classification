"""
Tests for the synthetic data generating process.

Tests the SCM-based data generation:
- Output format and shape
- Reproducibility with seeds
- Basic sanity checks on the generated data
"""

import numpy as np
import pandas as pd

from invariance_for_classification.generate_data.synthetic_DGP import (
    _simple_scm_one_env,
    generate_scm_data,
)


class TestGenerateScmData:
    """Tests for generate_scm_data function."""

    def test_output_is_dataframe(self):
        """Should return a pandas DataFrame."""
        df = generate_scm_data(n_per_env=100, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        """Should have expected columns."""
        df = generate_scm_data(n_per_env=100, seed=42)
        expected_cols = {"X1", "X2", "X3", "Y", "E"}
        assert set(df.columns) == expected_cols

    def test_sample_size(self):
        """Total samples = n_per_env * n_environments."""
        n_per_env = 100
        int_vals = [-1.0, 0.0, 1.0]
        df = generate_scm_data(n_per_env=n_per_env, int_vals=int_vals, seed=42)
        assert len(df) == n_per_env * len(int_vals)

    def test_default_environments(self):
        """Default should have 5 environments."""
        df = generate_scm_data(n_per_env=50, seed=42)
        assert df["E"].nunique() == 5
        assert len(df) == 50 * 5

    def test_y_is_binary(self):
        """Y should be binary (0 or 1)."""
        df = generate_scm_data(n_per_env=200, seed=42)
        assert set(df["Y"].unique()).issubset({0, 1})

    def test_environment_labels_correct(self):
        """Environment labels should be 0, 1, ..., n_envs-1."""
        int_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
        df = generate_scm_data(n_per_env=50, int_vals=int_vals, seed=42)
        assert set(df["E"].unique()) == set(range(len(int_vals)))

    def test_reproducibility_with_seed(self):
        """Same seed should give same data."""
        df1 = generate_scm_data(n_per_env=100, seed=123)
        df2 = generate_scm_data(n_per_env=100, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds should give different data."""
        df1 = generate_scm_data(n_per_env=100, seed=1)
        df2 = generate_scm_data(n_per_env=100, seed=2)
        # at least one value should differ
        assert not df1.equals(df2)

    def test_return_int_values(self):
        """Test return_int_values option."""
        df, int_df = generate_scm_data(
            n_per_env=50, int_vals=[-1.0, 1.0], seed=42, return_int_values=True
        )
        assert isinstance(int_df, pd.DataFrame)
        assert "int_value" in int_df.columns
        assert len(int_df) == len(df)

    def test_int_values_match_environments(self):
        """Intervention values should match the environment structure."""
        int_vals = [-2.0, 0.0, 2.0]
        n_per_env = 100
        df, int_df = generate_scm_data(
            n_per_env=n_per_env, int_vals=int_vals, seed=42, return_int_values=True
        )

        for env_idx, int_val in enumerate(int_vals):
            env_mask = df["E"] == env_idx
            assert (int_df.loc[env_mask, "int_value"] == int_val).all()


class TestSimpleScmOneEnv:
    """Tests for the internal _simple_scm_one_env function."""

    def test_output_shapes(self):
        """Should return DataFrames with correct shapes."""
        rng = np.random.default_rng(42)
        n = 100
        df, int_df = _simple_scm_one_env(n, int_val=1.0, env_idx=0, rng=rng)

        assert len(df) == n
        assert len(int_df) == n
        assert set(df.columns) == {"X1", "X2", "X3", "Y", "E"}
        assert set(int_df.columns) == {"int_value"}

    def test_environment_index_correct(self):
        """Environment index should be constant."""
        rng = np.random.default_rng(42)
        env_idx = 3
        df, _ = _simple_scm_one_env(n=50, int_val=0.0, env_idx=env_idx, rng=rng)
        assert (df["E"] == env_idx).all()

    def test_int_value_correct(self):
        """Intervention value should be constant."""
        rng = np.random.default_rng(42)
        int_val = 2.5
        _, int_df = _simple_scm_one_env(n=50, int_val=int_val, env_idx=0, rng=rng)
        assert (int_df["int_value"] == int_val).all()


class TestDgpSanityChecks:
    """Sanity checks that the DGP generates reasonable data."""

    def test_x1_varies_with_intervention(self):
        """X1 should vary systematically with intervention value."""
        df = generate_scm_data(n_per_env=500, int_vals=[-2.0, 2.0], seed=42)

        mean_x1_low = df[df["E"] == 0]["X1"].mean()
        mean_x1_high = df[df["E"] == 1]["X1"].mean()

        # int_val affects X1: X1 = alpha * int_val + noise
        # alpha = -1, so higher int_val -> lower X1
        assert mean_x1_high < mean_x1_low

    def test_y_depends_on_x1(self):
        """Y should depend on X1."""
        df = generate_scm_data(n_per_env=1000, seed=42)

        # higher X1 -> higher P(Y=1) approximately
        median_x1 = df["X1"].median()
        p_y_low_x1 = df[df["X1"] < median_x1]["Y"].mean()
        p_y_high_x1 = df[df["X1"] >= median_x1]["Y"].mean()

        assert p_y_high_x1 > p_y_low_x1

    def test_x3_correlates_with_y(self):
        """X3 should correlate with Y (X3 = Y + noise)."""
        df = generate_scm_data(n_per_env=500, seed=42)

        corr = df["X3"].corr(df["Y"])
        assert corr > 0.3  # should have positive correlation
