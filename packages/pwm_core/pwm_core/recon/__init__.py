"""Reconstruction solvers for PWM imaging modalities.

This module provides classical and modern reconstruction algorithms
for various imaging inverse problems.

Solvers by modality:
- Widefield/Confocal: Richardson-Lucy, PnP
- CT: FBP, SART
- MRI: ESPIRiT, SENSE, CS-MRI
- SPC: TVAL3, ISTA/FISTA
- CASSI/CACTI: GAP-TV
- SIM: Wiener-SIM
- Ptychography: ePIE
- Holography: Angular Spectrum
"""

from pwm_core.recon.portfolio import run_portfolio
from pwm_core.recon.pnp import run_pnp, pnp_hqs, pnp_admm, pnp_fista, get_denoiser

# Richardson-Lucy
try:
    from pwm_core.recon.richardson_lucy import (
        richardson_lucy_2d,
        richardson_lucy_3d,
        richardson_lucy_operator,
        run_richardson_lucy,
    )
except ImportError:
    richardson_lucy_2d = richardson_lucy_3d = None
    richardson_lucy_operator = run_richardson_lucy = None

# CT solvers
try:
    from pwm_core.recon.ct_solvers import (
        fbp_2d,
        sart_2d,
        sart_operator,
        run_fbp,
        run_sart,
    )
except ImportError:
    fbp_2d = sart_2d = sart_operator = None
    run_fbp = run_sart = None

# CS solvers
try:
    from pwm_core.recon.cs_solvers import (
        tval3,
        ista,
        fista,
        tv_prox_2d,
        run_tval3,
        run_ista,
    )
except ImportError:
    tval3 = ista = fista = tv_prox_2d = None
    run_tval3 = run_ista = None

# GAP-TV
try:
    from pwm_core.recon.gap_tv import (
        gap_tv_cassi,
        gap_tv_cacti,
        gap_tv_operator,
        run_gap_tv,
        run_gap_denoise,
    )
except ImportError:
    gap_tv_cassi = gap_tv_cacti = gap_tv_operator = None
    run_gap_tv = run_gap_denoise = None

# MRI solvers
try:
    from pwm_core.recon.mri_solvers import (
        estimate_sensitivity_maps,
        sense_reconstruction,
        cs_mri_wavelet,
        zero_filled_reconstruction,
        run_espirit_recon,
        run_cs_mri,
    )
except ImportError:
    estimate_sensitivity_maps = sense_reconstruction = None
    cs_mri_wavelet = zero_filled_reconstruction = None
    run_espirit_recon = run_cs_mri = None

# SIM solver
try:
    from pwm_core.recon.sim_solver import (
        wiener_sim_2d,
        fairsim_reconstruction,
        run_sim_reconstruction,
    )
except ImportError:
    wiener_sim_2d = fairsim_reconstruction = None
    run_sim_reconstruction = None

# Ptychography
try:
    from pwm_core.recon.ptychography_solver import (
        epie,
        pie,
        create_probe,
        run_epie,
    )
except ImportError:
    epie = pie = create_probe = run_epie = None

# Holography
try:
    from pwm_core.recon.holography_solver import (
        angular_spectrum_propagate,
        extract_plus_one_order,
        reconstruct_offaxis_hologram,
        run_holography_reconstruction,
    )
except ImportError:
    angular_spectrum_propagate = extract_plus_one_order = None
    reconstruct_offaxis_hologram = run_holography_reconstruction = None

__all__ = [
    # Portfolio
    "run_portfolio",
    # PnP
    "run_pnp",
    "pnp_hqs",
    "pnp_admm",
    "pnp_fista",
    "get_denoiser",
    # Richardson-Lucy
    "richardson_lucy_2d",
    "richardson_lucy_3d",
    "richardson_lucy_operator",
    "run_richardson_lucy",
    # CT
    "fbp_2d",
    "sart_2d",
    "sart_operator",
    "run_fbp",
    "run_sart",
    # CS
    "tval3",
    "ista",
    "fista",
    "tv_prox_2d",
    "run_tval3",
    "run_ista",
    # GAP-TV
    "gap_tv_cassi",
    "gap_tv_cacti",
    "gap_tv_operator",
    "run_gap_tv",
    "run_gap_denoise",
    # MRI
    "estimate_sensitivity_maps",
    "sense_reconstruction",
    "cs_mri_wavelet",
    "zero_filled_reconstruction",
    "run_espirit_recon",
    "run_cs_mri",
    # SIM
    "wiener_sim_2d",
    "fairsim_reconstruction",
    "run_sim_reconstruction",
    # Ptychography
    "epie",
    "pie",
    "create_probe",
    "run_epie",
    # Holography
    "angular_spectrum_propagate",
    "extract_plus_one_order",
    "reconstruct_offaxis_hologram",
    "run_holography_reconstruction",
]
