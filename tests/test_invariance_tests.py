"""
Tests for invariance tests implementations.

Tests the individual invariance tests:
- InvariantResidualDistributionTest
- DeLongTest
- TramGcmTest
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from invariance_for_classification import StabilizedClassificationClassifier
from invariance_for_classification.generate_data.synthetic_DGP import generate_scm_data
from invariance_for_classification.invariance_tests import (
    DeLongTest,
    InvariantResidualDistributionTest,
    TramGcmTest,
)

# --- Test fixtures ---


@pytest.fixture
def synthetic_data():
    """Generate synthetic data from the SCM."""
    df = generate_scm_data(n_per_env=200, seed=42)
    X = df[["X1", "X2", "X3"]].values
    y = df["Y"].values
    E = df["E"].values
    return X, y, E


@pytest.fixture
def invariant_data():
    """Generate data where X is invariant (independent of Y given E)."""
    rng = np.random.default_rng(42)
    n = 500
    E = rng.choice([0, 1, 2], size=n)
    # X1 is a direct cause of Y, invariant
    X1 = rng.normal(size=n)
    y = (X1 + rng.logistic(size=n) > 0).astype(int)
    X = X1.reshape(-1, 1)
    return X, y, E


@pytest.fixture
def non_invariant_data():
    """Generate data where E directly affects Y (non-invariant empty set)."""
    rng = np.random.default_rng(123)
    n = 500
    E = rng.choice([0, 1, 2], size=n)
    # Y depends directly on E
    y = ((E - 1) + rng.logistic(size=n) > 0).astype(int)
    X = rng.normal(size=(n, 2))
    return X, y, E


class TestInvariantResidualDistributionTest:
    """Tests for the InvariantResidualDistributionTest."""

    @pytest.mark.parametrize("clf_type", ["RF", "LR"])
    def test_p_value_bounds(self, synthetic_data, clf_type):
        """P-values should be in [0, 1]."""
        X, y, E = synthetic_data
        test = InvariantResidualDistributionTest(test_classifier_type=clf_type)
        p_val = test.test(X, y, E)
        assert 0 <= p_val <= 1

    def test_single_env_returns_one(self, synthetic_data):
        """With a single environment, p-value should be 1.0."""
        X, y, _ = synthetic_data
        E_single = np.zeros(len(y))
        test = InvariantResidualDistributionTest()
        p_val = test.test(X, y, E_single)
        assert p_val == 1.0

    def test_empty_feature_set(self, synthetic_data):
        """Test with empty feature set (checking if P(Y) varies across E)."""
        _, y, E = synthetic_data
        X_empty = np.zeros((len(y), 0))
        test = InvariantResidualDistributionTest()
        p_val = test.test(X_empty, y, E)
        assert 0 <= p_val <= 1

    def test_invariant_predictor_high_pvalue(self, invariant_data):
        """Invariant predictors should have high p-values."""
        X, y, E = invariant_data
        test = InvariantResidualDistributionTest()
        p_val = test.test(X, y, E)
        # should not reject at alpha=0.05
        assert p_val > 0.01

    @pytest.mark.parametrize("clf_type", ["RF", "LR"])
    def test_deterministic_with_seed(self, synthetic_data, clf_type):
        """Results should be reproducible with same random state."""
        X, y, E = synthetic_data
        test = InvariantResidualDistributionTest(test_classifier_type=clf_type)
        p1 = test.test(X, y, E)
        p2 = test.test(X, y, E)
        assert p1 == p2

    def test_invalid_classifier_type(self):
        """Should raise error for unknown classifier type."""
        with pytest.raises(ValueError, match="Unknown test_classifier_type"):
            InvariantResidualDistributionTest(test_classifier_type="invalid")


class TestDeLongTest:
    """Tests for the DeLong invariance test."""

    @pytest.mark.parametrize("clf_type", ["RF", "LR"])
    def test_p_value_bounds(self, synthetic_data, clf_type):
        """P-values should be in [0, 1]."""
        X, y, E = synthetic_data
        test = DeLongTest(test_classifier_type=clf_type)
        p_val = test.test(X, y, E)
        assert 0 <= p_val <= 1

    def test_single_env_returns_one(self, synthetic_data):
        """With single environment, p-value should be 1.0."""
        X, y, _ = synthetic_data
        E_single = np.zeros(len(y))
        test = DeLongTest()
        p_val = test.test(X, y, E_single)
        assert p_val == 1.0

    def test_empty_feature_set(self, synthetic_data):
        """Test with empty feature set."""
        _, y, E = synthetic_data
        X_empty = np.zeros((len(y), 0))
        test = DeLongTest()
        p_val = test.test(X_empty, y, E)
        assert 0 <= p_val <= 1

    def test_invariant_predictor(self, invariant_data):
        """Invariant predictors should have high p-values."""
        X, y, E = invariant_data
        test = DeLongTest()
        p_val = test.test(X, y, E)
        # should not reject at alpha=0.05
        assert p_val > 0.01


class TestTramGcmTest:
    """Tests for the Tram-GCM invariance test."""

    @pytest.mark.parametrize("clf_type", ["RF", "LR"])
    def test_p_value_bounds(self, synthetic_data, clf_type):
        """P-values should be in [0, 1]."""
        X, y, E = synthetic_data
        test = TramGcmTest(test_classifier_type=clf_type)
        p_val = test.test(X, y, E)
        assert 0 <= p_val <= 1

    def test_single_env_returns_one(self, synthetic_data):
        """With single environment, p-value should be 1.0."""
        X, y, _ = synthetic_data
        E_single = np.zeros(len(y))
        test = TramGcmTest()
        p_val = test.test(X, y, E_single)
        assert p_val == 1.0

    def test_empty_feature_set(self, synthetic_data):
        """Test with empty feature set."""
        _, y, E = synthetic_data
        X_empty = np.zeros((len(y), 0))
        test = TramGcmTest()
        p_val = test.test(X_empty, y, E)
        assert 0 <= p_val <= 1

    def test_invariant_predictor(self, invariant_data):
        """Invariant predictors should have high p-values."""
        X, y, E = invariant_data
        test = TramGcmTest()
        p_val = test.test(X, y, E)
        # should not reject
        assert p_val > 0.01

    def test_invalid_classifier_type(self):
        """Should raise error for unknown classifier type."""
        test = TramGcmTest(test_classifier_type="invalid")
        X = np.random.rand(50, 2)
        y = np.random.randint(0, 2, 50)
        E = np.random.randint(0, 2, 50)
        with pytest.raises(ValueError, match="Unknown test_classifier_type"):
            test.test(X, y, E)


class TestInvarianceTestsComparison:
    """Compare behavior across different invariance tests."""

    def test_all_tests_agree_on_single_env(self, synthetic_data):
        """All tests should return 1.0 for single environment."""
        X, y, _ = synthetic_data
        E_single = np.zeros(len(y))

        tests = [
            InvariantResidualDistributionTest(),
            DeLongTest(),
            TramGcmTest(),
        ]
        for test in tests:
            assert test.test(X, y, E_single) == 1.0

    def test_all_tests_produce_valid_pvalues(self, synthetic_data):
        """All tests should produce p-values in [0, 1]."""
        X, y, E = synthetic_data

        tests = [
            InvariantResidualDistributionTest(),
            DeLongTest(),
            TramGcmTest(),
        ]
        for test in tests:
            p_val = test.test(X, y, E)
            assert 0 <= p_val <= 1, (
                f"{test.__class__.__name__} produced invalid p-value"
            )


class TestFindsInvariantSubsets:
    """
    Test if the invariance tests correctly identify invariant subsets.

    In the synthetic SCM: E -> X1 -> Y -> X2 <- E, and Y -> X3.
    - X1 should be invariant (Y | X1 is invariant).
    - {X1, X3} should also be invariant (Y | X1, X3 is invariant).
    - X2 is not invariant (Y | X2 depends on E).
    """

    @pytest.fixture
    def large_data(self):
        """Larger dataset for statistical power."""
        df = generate_scm_data(n_per_env=2000, seed=42)
        return df

    @pytest.mark.parametrize(
        "inv_test_cls",
        [InvariantResidualDistributionTest, DeLongTest, TramGcmTest],
    )
    @pytest.mark.parametrize("pred_classifier_type", ["RF", "LR"])
    @pytest.mark.parametrize("test_classifier_type", ["RF", "LR"])
    def test_finds_correct_invariant_subsets(
        self, large_data, inv_test_cls, pred_classifier_type, test_classifier_type
    ):
        """Test that invariance tests identify the correct invariant subsets."""
        df = large_data

        clf = StabilizedClassificationClassifier(
            alpha_inv=0.05, n_bootstrap=20, random_state=42
        )
        X, y, environment = clf._validate_input(df, y="Y", environment="E")

        # Initialize label encoder (needed by _find_invariant_subsets)
        clf.le_ = LabelEncoder()
        y = clf.le_.fit_transform(y)
        clf.classes_ = clf.le_.classes_
        n_features = X.shape[1]

        # Set up prediction classifier
        if pred_classifier_type == "RF":
            pred_classifier = RandomForestClassifier(
                n_estimators=100, oob_score=True, random_state=42, n_jobs=1
            )
        else:
            pred_classifier = LogisticRegression(random_state=42)

        # Instantiate the invariance test
        inv_test = inv_test_cls(test_classifier_type=test_classifier_type)

        subset_stats, _ = clf._find_invariant_subsets(
            X, y, environment, n_features, inv_test, pred_classifier
        )

        invariant_subsets = {frozenset(s["subset"]) for s in subset_stats}

        # {X1} (index 0) is invariant; {X1,X3} should also be invariant in this SCM
        # Note: DeLong test is an indirect test that may not have valid level
        # in all situations, so we only require {0} for it
        if inv_test.name == "delong":
            expected = {frozenset({0})}
        else:
            expected = {frozenset({0}), frozenset({0, 2})}

        missing = expected - invariant_subsets
        assert not missing, (
            f"Expected invariant subsets {[set(e) for e in expected]}. "
            f"Missing: {sorted([set(m) for m in missing])}. "
            f"Found: {sorted([set(s) for s in invariant_subsets])}"
        )
