"""Data generation utilities for invariance-for-classification."""

from invariance_for_classification.generate_data.complex_DGP import (
    generate_complex_scm_data,
)
from invariance_for_classification.generate_data.synthetic_DGP import generate_scm_data

__all__ = [
    "generate_scm_data",
    "generate_complex_scm_data",
]
