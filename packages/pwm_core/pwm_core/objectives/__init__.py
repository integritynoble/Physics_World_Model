"""pwm_core.objectives
======================

Likelihood-aware objective functions for reconstruction and calibration.

Modules
-------
base    NegLogLikelihood ABC + 6 implementations + OBJECTIVE_REGISTRY
prior   PriorSpec for regularization terms (TV, L1-wavelet, low-rank, deep prior)
"""

from pwm_core.objectives.base import (
    ObjectiveSpec,
    NegLogLikelihood,
    PoissonNLL,
    GaussianNLL,
    ComplexGaussianNLL,
    MixedPoissonGaussianNLL,
    HuberNLL,
    TukeyBiweightNLL,
    OBJECTIVE_REGISTRY,
    build_objective,
)
from pwm_core.objectives.prior import PriorSpec

__all__ = [
    "ObjectiveSpec",
    "NegLogLikelihood",
    "PoissonNLL",
    "GaussianNLL",
    "ComplexGaussianNLL",
    "MixedPoissonGaussianNLL",
    "HuberNLL",
    "TukeyBiweightNLL",
    "OBJECTIVE_REGISTRY",
    "build_objective",
    "PriorSpec",
]
