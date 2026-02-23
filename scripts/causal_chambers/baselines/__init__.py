"""
Baseline methods for comparison with stabilized classification.

not part of the main package => exist only for benchmarking
"""

from .icp import ICPglmClassifier, ICPrfClassifier
from .irm_2 import IRM2Classifier
from .neural_net import ERMClassifier, IRMClassifier, VRExClassifier

__all__ = [
    "IRMClassifier",
    "IRM2Classifier",
    "VRExClassifier",
    "ERMClassifier",
    "ICPglmClassifier",
    "ICPrfClassifier",
]
