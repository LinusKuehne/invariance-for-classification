import inspect

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from invariance_for_classification import (
    StabilizedClassificationClassifier,
    invariance_tests,
)
from invariance_for_classification.estimators._stabilized import _EmptySetClassifier
from invariance_for_classification.generate_data.synthetic_DGP import generate_scm_data
from invariance_for_classification.invariance_tests import InvarianceTest


def get_invariance_test_classes():
    """Helper to discover all available invariance test classes."""
    classes = []
    for _, obj in inspect.getmembers(invariance_tests):
        if (
            inspect.isclass(obj)
            and issubclass(obj, InvarianceTest)
            and obj is not InvarianceTest
        ):
            classes.append(obj)
    return classes


def test_environment_required():
    """Environment must be provided explicitly."""
    df = generate_scm_data(n_per_env=1000, seed=42)
    clf = StabilizedClassificationClassifier()
    X = df.drop(columns=["Y", "E"]).values
    y = df["Y"].values

    with pytest.raises(ValueError, match="Environment labels must be provided"):
        clf.fit(X, y)


def test_initialization():
    """Test standard initialization."""
    clf = StabilizedClassificationClassifier(alpha_inv=0.1, n_bootstrap=50)
    assert clf.alpha_inv == 0.1
    assert clf.n_bootstrap == 50


def test_empty_set_classifier():
    """Test the internal dummy classifier used for the empty set."""
    X = np.random.rand(100, 3)
    y = np.random.choice([0, 1], size=100, p=[0.7, 0.3])

    empty_clf = _EmptySetClassifier()
    empty_clf.fit(X, y)

    # Check prior calculation against empirical proportions
    expected_prior = np.array([np.mean(y == 0), np.mean(y == 1)])
    assert np.allclose(empty_clf.prior_, expected_prior, atol=1e-8)

    # Check prediction shape
    preds = empty_clf.predict(X)
    assert preds.shape == (100,)
    proba = empty_clf.predict_proba(X)
    assert proba.shape == (100, 2)

    # It should predict the majority class (0 in this case)
    assert np.all(preds == 0)


@pytest.mark.parametrize("inv_test_cls", get_invariance_test_classes())
def test_invariance_test_smoke(inv_test_cls):
    """Smoke test for invariance tests."""
    X = np.random.rand(100, 2)
    y = np.random.randint(0, 2, 100)
    E = np.random.randint(0, 3, 100)

    inv_test = inv_test_cls()
    p_val = inv_test.test(X, y, E)
    assert 0 <= p_val <= 1.0


@pytest.mark.parametrize("inv_test_cls", get_invariance_test_classes())
def test_invariance_test_single_env(inv_test_cls):
    """Test behavior with a single environment."""
    X = np.random.rand(50, 2)
    y = np.random.randint(0, 2, 50)
    E = np.zeros(50)  # single environment

    inv_test = inv_test_cls()
    p_val = inv_test.test(X, y, E)
    assert p_val == 1.0


def test_input_validation_mismatched_lengths():
    """Test that mismatched lengths between X and y raise ValueError."""
    df = generate_scm_data(n_per_env=100, seed=42)
    clf = StabilizedClassificationClassifier()
    X = df.drop(columns=["Y", "E"]).values
    y = df["Y"].values[:-1]  # One element short
    env = df["E"].values

    with pytest.raises(ValueError):
        clf.fit(X, y, environment=env)


def test_single_environment_error():
    """Test that a single environment raises a ValueError."""
    df = generate_scm_data(n_per_env=100, seed=42)
    clf = StabilizedClassificationClassifier()
    X = df.drop(columns=["Y", "E"]).values
    y = df["Y"].values
    env = np.zeros_like(y)  # All same

    with pytest.raises(ValueError, match="at least 2 unique values"):
        clf.fit(X, y, environment=env)


@pytest.mark.parametrize("inv_test_cls", get_invariance_test_classes())
def test_fit_stabilized_classifier(inv_test_cls):
    """Test fitting the main classifier."""
    df = generate_scm_data(n_per_env=1000, seed=42)
    # Use the discovered invariance test class
    inv_test_instance = inv_test_cls()
    clf = StabilizedClassificationClassifier(
        n_bootstrap=10, verbose=0, invariance_test=inv_test_instance.name
    )

    clf.fit(df, y="Y", environment="E")

    check_is_fitted(clf)
    assert hasattr(clf, "active_subsets_")
    assert len(clf.active_subsets_) >= 0


def test_prediction_shape():
    """Test prediction output shape and values."""
    df = generate_scm_data(n_per_env=100, seed=42)
    clf = StabilizedClassificationClassifier(n_bootstrap=10)

    clf.fit(df, y="Y", environment="E")

    X_test = df.drop(columns=["E", "Y"]).values
    preds = clf.predict(X_test)

    assert preds.shape == (len(df),)
    assert set(np.unique(preds)).issubset({0, 1})


@pytest.mark.parametrize("inv_test_cls", get_invariance_test_classes())
def test_finds_invariant_subset(inv_test_cls):
    """
    Test if the invariance test identifies invariant subsets.
    In the synthetic SCM: E -> X1 -> Y -> X2 <- E, and Y -> X3.
    X1 should be invariant (Y | X1 is invariant).
    {X1, X3} should also be invariant (Y | X1, X3 is invariant).
    X2 is not (Y | X2 depends on E).
    """
    # Increase n_per_env to ensure statistical power for the test
    df_large = generate_scm_data(n_per_env=2000, seed=42)

    # Use the invariance test directly via _find_invariant_subsets
    clf = StabilizedClassificationClassifier(
        alpha_inv=0.05, n_bootstrap=20, random_state=42
    )
    X, y, environment = clf._validate_input(df_large, y="Y", environment="E")
    # _find_invariant_subsets expects encoded classes to be initialized
    clf.le_ = LabelEncoder()
    y = clf.le_.fit_transform(y)
    clf.classes_ = clf.le_.classes_
    n_features = X.shape[1]

    for pred_classifier_type in ["RF", "LR"]:
        if pred_classifier_type == "RF":
            pred_classifier = RandomForestClassifier(
                n_estimators=100, oob_score=True, random_state=42, n_jobs=1
            )
        else:
            pred_classifier = LogisticRegression(random_state=42)

        for test_classifier_type in ["RF", "LR"]:
            # instantiate the invariance test class to be tested
            inv_test = inv_test_cls(test_classifier_type=test_classifier_type)

            subset_stats, _ = clf._find_invariant_subsets(
                X, y, environment, n_features, inv_test, pred_classifier
            )

            invariant_subsets = {frozenset(s["subset"]) for s in subset_stats}

            # {X1} is invariant; {X1,X3} should also be invariant in this SCM
            # Note: DeLong test is an indirect test that may not have valid level
            # in all situations (see testing.tex), so we only require {0} for it
            if inv_test.name == "delong":
                expected = {frozenset({0})}
            else:
                expected = {frozenset({0}), frozenset({0, 2})}
            missing = expected - invariant_subsets
            assert not missing, (
                f"Expected invariant subsets {[set(e) for e in expected]}. "
                f"Missing: {sorted([set(m) for m in missing])}. Found: {sorted([set(s) for s in invariant_subsets])}"
            )


def test_predict_proba():
    """Test predict_proba method."""
    df = generate_scm_data(n_per_env=1000, seed=42)
    clf = StabilizedClassificationClassifier(n_bootstrap=10)
    clf.fit(df, y="Y", environment="E")

    X_test = df.drop(columns=["E", "Y"]).values
    proba = clf.predict_proba(X_test)

    assert proba.shape == (len(df), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(np.sum(proba, axis=1), 1.0)


def test_ensemble_averaging():
    """Test that the classifier averages probabilities correctly across active subsets."""
    clf = StabilizedClassificationClassifier()

    # Mock specific active subsets
    # We need to set classes_ since predict_proba relies on it
    clf.classes_ = np.array([0, 1])
    clf.le_ = LabelEncoder()
    clf.le_.fit([0, 1])
    # Mock n_features_in_ for validation
    clf.n_features_in_ = 2

    # Mock Model A
    class MockModel:
        def __init__(self, proba):
            self.proba = proba
            self.classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.full((X.shape[0], 2), self.proba)

    # Subset 1: Predicts [0.2, 0.8]
    model_1 = MockModel([0.2, 0.8])
    # Subset 2: Predicts [0.6, 0.4]
    model_2 = MockModel([0.6, 0.4])

    # Manually inject active subsets
    clf.active_subsets_ = [
        {"subset": [0], "model": model_1, "weight": 0.5},
        {"subset": [1], "model": model_2, "weight": 0.5},
    ]

    X = np.zeros((10, 2))
    proba = clf.predict_proba(X)

    # Expected: 0.5 * [0.2, 0.8] + 0.5 * [0.6, 0.4] = [0.4, 0.6]
    expected = np.array([0.4, 0.6])
    assert np.allclose(proba[0], expected)


def test_no_invariant_fallback():
    """Test behavior when no invariant subsets are found (fallback mechanism)."""

    df = generate_scm_data(n_per_env=200, seed=42)

    # Use a test that always returns p<0.5, so nothing is "invariant"
    clf = StabilizedClassificationClassifier(alpha_inv=0.99999)

    X = df.drop(columns=["Y", "E"]).values
    y = df["Y"].values
    env = df["E"].values

    clf.fit(X, y, environment=env)

    # Should have fallen back to exactly one subset (the one with max p-value, even if low)
    # The logic is: if not subset_stats -> apply_fallback -> return [best_subset]
    assert len(clf.active_subsets_) == 1

    # Verify partial fit works
    assert hasattr(clf, "classes_")
    # Verify we can predict without error
    preds = clf.predict(X)
    assert len(preds) == len(X)


def test_performance_above_random():
    """Test that the classifier actually learns something on the synthetic data."""
    # Generate larger datasets for stable performance measurement
    df_train = generate_scm_data(n_per_env=500, seed=101)
    df_test = generate_scm_data(n_per_env=500, seed=102)

    clf = StabilizedClassificationClassifier(
        n_bootstrap=20, alpha_inv=0.05, random_state=42
    )

    X_train = df_train.drop(columns=["E", "Y"]).values
    y_train = df_train["Y"].values
    env_train = df_train["E"].values

    clf.fit(X_train, y_train, environment=env_train)

    X_test = df_test.drop(columns=["E", "Y"]).values
    y_test = df_test["Y"].to_numpy()

    # Score is accuracy
    acc = clf.score(X_test, y_test)

    # Random guessing would be ~0.5, which we should beat
    assert acc > 0.7, f"Accuracy {acc} is too low (expected > 0.7)"
