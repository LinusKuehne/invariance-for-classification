import numpy as np
from sklearn.linear_model import LogisticRegression

from invariance_for_classification import StabilizedClassificationClassifier
from invariance_for_classification.invariance_tests import (
    InvariantResidualDistributionTest,
)


def test_invariance_test_lr():
    """Test invariance test with LR"""
    test = InvariantResidualDistributionTest(classifier_type="LR")
    assert isinstance(test.estimator, LogisticRegression)

    X = np.random.rand(20, 2)
    y = np.random.randint(0, 2, 20)
    E = np.random.randint(0, 2, 20)

    p_val = test.test(X, y, E)
    assert 0 <= p_val <= 1.0


def test_stabilized_lr_integration():
    """Integration test for StabilizedClassificationClassifier with LR"""
    # Create simple dataset
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 3)
    # y depends on X[:, 0] only
    y = (X[:, 0] > 0).astype(int)
    # Env depends on X[:, 1]
    E = (X[:, 1] > 0).astype(int)

    clf = StabilizedClassificationClassifier(
        classifier_type="LR", n_bootstrap=10, verbose=1
    )
    clf.fit(X, y, environment=E)

    # Check predictions
    pred = clf.predict(X)
    assert pred.shape == (n,)

    proba = clf.predict_proba(X)
    assert proba.shape == (n, 2)

    # Check if we have active subsets and they are LR (unless empty set)
    # With this data, X[:,0] should be invariant and predictive
    assert len(clf.active_subsets_) > 0
    found_lr = False
    for stat in clf.active_subsets_:
        if hasattr(
            stat["model"], "coef_"
        ):  # Check for LR characteristics or isinstance
            if isinstance(stat["model"], LogisticRegression):
                found_lr = True
                break

    # It might be that empty set is also included or selected
    if not found_lr:
        # If only empty set was selected, check if we at least configured it right?
        # But for this dataset, X[:,0] should be better than empty.
        pass

    assert found_lr or any(
        isinstance(s["model"], LogisticRegression) for s in clf.active_subsets_
    )
