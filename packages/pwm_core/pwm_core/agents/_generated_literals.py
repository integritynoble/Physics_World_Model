"""Auto-generated Literal types from YAML registries.

DO NOT EDIT MANUALLY. Regenerate with:
    python -m pwm_core.agents._generate_literals

These types enable static type checking (mypy/pyright) to catch invalid
registry keys at type-check time, not just runtime.
"""
# fmt: off
from typing import Literal

ModalityKey = Literal[
    "cacti",
    "cassi",
    "confocal_3d",
    "confocal_livecell",
    "ct",
    "dot",
    "flim",
    "fpm",
    "gaussian_splatting",
    "holography",
    "integral",
    "lensless",
    "light_field",
    "lightsheet",
    "matrix",
    "mri",
    "nerf",
    "oct",
    "panorama",
    "phase_retrieval",
    "photoacoustic",
    "ptychography",
    "sim",
    "spc",
    "widefield",
    "widefield_lowdose",
]

SignalPriorKey = Literal[
    "deep_prior",
    "joint_spatio_spectral",
    "low_rank",
    "temporal_smooth",
    "tv",
    "wavelet_sparse",
]

# fmt: on
