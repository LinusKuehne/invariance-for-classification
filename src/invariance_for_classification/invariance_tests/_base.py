from abc import ABC, abstractmethod

import numpy as np


class InvarianceTest(ABC):
    """
    Abstract base class for invariance tests.

    An invariance test checks the null hypothesis that the conditional distribution
    of the target Y given a subset of predictors X_S is invariant across environments E.

    H0: Y | X_S, E = Y | X_S
    """

    @abstractmethod
    def test(self, X: np.ndarray, y: np.ndarray, E: np.ndarray) -> float:
        """
        Perform invariance test and return a p-value.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            subset of predictors to test
        y : np.ndarray of shape (n_samples,)
            target variable.
        E : np.ndarray of shape (n_samples,)
            environment indicator

        Returns
        -------
        p_value : float
            The p-value of the test. High p-values indicate that null hypothesis
            (invariance) cannot be rejected.
        """
        raise NotImplementedError("Subclasses must implement test().")
