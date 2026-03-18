"""pwm_core.physics.electron
=============================

Pure-function helpers for electron microscopy modalities (SEM, TEM, ET).
All functions are stateless and operate on numpy arrays (D4 rule).
"""

from pwm_core.physics.electron.sem_helpers import (
    se_yield,
    bse_yield,
    apply_scan_drift,
)
from pwm_core.physics.electron.tem_helpers import (
    compute_ctf,
    phase_object_transmission,
    ctf_zeros,
)
from pwm_core.physics.electron.et_helpers import (
    tilt_project,
    alignment_shift,
    sirt_recon,
)

__all__ = [
    "se_yield",
    "bse_yield",
    "apply_scan_drift",
    "compute_ctf",
    "phase_object_transmission",
    "ctf_zeros",
    "tilt_project",
    "alignment_shift",
    "sirt_recon",
]
