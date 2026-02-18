"""Mismatch simulation and calibration for operator correction.

This module provides functions to:
1. Simulate imperfect/mismatched forward operators
2. Calibrate operator parameters from measurements
3. Reconstruct with corrected operators

Supported modalities:
- CT: Center of rotation
- MRI: Coil sensitivities
- CASSI: Dispersion step
- CACTI: Mask timing
- SPC: Gain/bias
- Lensless: PSF shift
- Ptychography: Position offset
"""

from pwm_core.mismatch.operators import (
    # Utility
    compute_psnr,
    # CT
    ct_radon_forward,
    ct_sart_tv_recon,
    ct_calibrate_cor,
    # MRI
    mri_generate_coil_sensitivities,
    mri_forward_sense,
    mri_sense_recon,
    mri_estimate_sensitivities_acs,
    # CASSI
    cassi_shift,
    cassi_shift_back,
    cassi_forward,
    cassi_adjoint,
    cassi_gap_denoise,
    cassi_calibrate_step,
    # CACTI
    cacti_forward,
    cacti_gap_tv,
    cacti_calibrate_timing,
    # SPC
    spc_forward,
    spc_lsq_recon,
    spc_calibrate_gain_bias,
    # Lensless
    lensless_forward,
    lensless_admm_tv,
    lensless_calibrate_shift,
    # Ptychography
    ptycho_get_positions,
    ptycho_forward,
    ptycho_epie_recon,
    ptycho_calibrate_offset,
)

__all__ = [
    # Utility
    "compute_psnr",
    # CT
    "ct_radon_forward",
    "ct_sart_tv_recon",
    "ct_calibrate_cor",
    # MRI
    "mri_generate_coil_sensitivities",
    "mri_forward_sense",
    "mri_sense_recon",
    "mri_estimate_sensitivities_acs",
    # CASSI
    "cassi_shift",
    "cassi_shift_back",
    "cassi_forward",
    "cassi_adjoint",
    "cassi_gap_denoise",
    "cassi_calibrate_step",
    # CACTI
    "cacti_forward",
    "cacti_gap_tv",
    "cacti_calibrate_timing",
    # SPC
    "spc_forward",
    "spc_lsq_recon",
    "spc_calibrate_gain_bias",
    # Lensless
    "lensless_forward",
    "lensless_admm_tv",
    "lensless_calibrate_shift",
    # Ptychography
    "ptycho_get_positions",
    "ptycho_forward",
    "ptycho_epie_recon",
    "ptycho_calibrate_offset",
]
