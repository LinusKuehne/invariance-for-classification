"""
Baseline methods for comparison with stabilized classification.

not part of the main package => exist only for benchmarking
"""

from .irm_linear import IRMLinearClassifier
from .irm_nn import IRMNNClassifier

__all__ = [
    "IRMNNClassifier",
    "IRMLinearClassifier",
]
