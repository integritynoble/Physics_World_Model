"""Reconstruction solvers for PWM imaging modalities.

This module provides classical and modern reconstruction algorithms
for various imaging inverse problems.

Solvers by modality:
- Widefield/Confocal: Richardson-Lucy, PnP, CARE, Noise2Void
- CT: FBP, SART, RED-CNN
- MRI: ESPIRiT, SENSE, CS-MRI, VarNet, MoDL
- SPC: TVAL3, ISTA/FISTA, PnP-FISTA, ISTA-Net, HATNet
- CASSI: MST (default), GAP-TV, HDNet, HSI-SDeCNN
- CACTI: GAP-TV, EfficientSCI, ELP-Unfolding
- SIM: Wiener-SIM, HiFi-SIM, DL-SIM
- Ptychography: ePIE, ML-ePIE, PtychoNN
- Holography: Angular Spectrum, TV-ADMM, PhaseNet
- Lensless: ADMM-TV, FlatNet
- Light-Sheet: Fourier Notch, VSNR, DeStripe
- NeRF: NeRF MLP, Instant-NGP
- Gaussian Splatting: 3DGS
- Matrix/Generic: FISTA-L2, LISTA, Diffusion Posterior
- Panorama/Multi-Focus: Laplacian Pyramid, Guided Filter, IFCNN
- Light Field: Shift-and-Sum, LFBM5D
- Integral Photography: Depth Estimation, DIBR
- Phase Retrieval / CDI: HIO, RAAR, Gerchberg-Saxton
- FLIM: Phasor Analysis, MLE Fit
- Photoacoustic: Back Projection, Time Reversal
- OCT: FFT Recon, Spectral Estimation
- FPM: Sequential Phase Retrieval, Gradient Descent
- DOT: Born/Tikhonov, L-BFGS-TV
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

# MST (CASSI default)
try:
    from pwm_core.recon.mst import (
        MST,
        mst_recon_cassi,
        shift_torch,
        shift_back_meas_torch,
    )
except ImportError:
    MST = mst_recon_cassi = None
    shift_torch = shift_back_meas_torch = None

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
        hifi_sim_2d,
        fairsim_reconstruction,
        run_sim_reconstruction,
    )
except ImportError:
    wiener_sim_2d = hifi_sim_2d = fairsim_reconstruction = None
    run_sim_reconstruction = None

# Ptychography
try:
    from pwm_core.recon.ptychography_solver import (
        epie,
        pie,
        ml_epie,
        create_probe,
        run_epie,
    )
except ImportError:
    epie = pie = ml_epie = create_probe = run_epie = None

# Holography
try:
    from pwm_core.recon.holography_solver import (
        angular_spectrum_propagate,
        extract_plus_one_order,
        reconstruct_offaxis_hologram,
        tv_holography_admm,
        run_holography_reconstruction,
    )
except ImportError:
    angular_spectrum_propagate = extract_plus_one_order = None
    reconstruct_offaxis_hologram = tv_holography_admm = None
    run_holography_reconstruction = None

# CARE U-Net (widefield, confocal)
try:
    from pwm_core.recon.care_unet import run_care
except ImportError:
    run_care = None

# Noise2Void (self-supervised denoising)
try:
    from pwm_core.recon.noise2void import run_noise2void
except ImportError:
    run_noise2void = None

# HDNet (CASSI)
try:
    from pwm_core.recon.hdnet import run_hdnet
except ImportError:
    run_hdnet = None

# HSI-SDeCNN (CASSI PnP)
try:
    from pwm_core.recon.hsi_sdecnn import run_hsi_sdecnn
except ImportError:
    run_hsi_sdecnn = None

# EfficientSCI (CACTI)
try:
    from pwm_core.recon.efficientsci import run_efficientsci
except ImportError:
    run_efficientsci = None

# ELP-Unfolding (CACTI)
try:
    from pwm_core.recon.elp_unfolding import run_elp_unfolding
except ImportError:
    run_elp_unfolding = None

# HATNet (SPC)
try:
    from pwm_core.recon.hatnet import run_hatnet
except ImportError:
    run_hatnet = None

# ISTA-Net (SPC)
try:
    from pwm_core.recon.ista_net import run_ista_net
except ImportError:
    run_ista_net = None

# RED-CNN (CT)
try:
    from pwm_core.recon.redcnn import run_redcnn
except ImportError:
    run_redcnn = None

# VarNet (MRI)
try:
    from pwm_core.recon.varnet import run_varnet
except ImportError:
    run_varnet = None

# MoDL (MRI)
try:
    from pwm_core.recon.modl import run_modl
except ImportError:
    run_modl = None

# PtychoNN (ptychography)
try:
    from pwm_core.recon.ptychonn import run_ptychonn
except ImportError:
    run_ptychonn = None

# PhaseNet (holography)
try:
    from pwm_core.recon.phasenet import run_phasenet
except ImportError:
    run_phasenet = None

# DL-SIM
try:
    from pwm_core.recon.dl_sim import run_dl_sim
except ImportError:
    run_dl_sim = None

# Lensless solver
try:
    from pwm_core.recon.lensless_solver import run_lensless
except ImportError:
    run_lensless = None

# FlatNet (lensless DL)
try:
    from pwm_core.recon.flatnet import run_flatnet
except ImportError:
    run_flatnet = None

# Light-sheet
try:
    from pwm_core.recon.lightsheet_solver import run_lightsheet
except ImportError:
    run_lightsheet = None

# DeStripe (light-sheet DL)
try:
    from pwm_core.recon.destripe_net import run_destripe
except ImportError:
    run_destripe = None

# LISTA (generic linear)
try:
    from pwm_core.recon.lista import run_lista
except ImportError:
    run_lista = None

# IFCNN (image fusion)
try:
    from pwm_core.recon.ifcnn import run_ifcnn
except ImportError:
    run_ifcnn = None

# Diffusion posterior sampling
try:
    from pwm_core.recon.diffusion_posterior import run_diffusion_posterior
except ImportError:
    run_diffusion_posterior = None

# NeRF
try:
    from pwm_core.recon.nerf_solver import run_nerf
except ImportError:
    run_nerf = None

# Gaussian Splatting
try:
    from pwm_core.recon.gaussian_splatting_solver import run_gaussian_splatting
except ImportError:
    run_gaussian_splatting = None

# Panorama / Multi-Focus Fusion
try:
    from pwm_core.recon.panorama_solver import (
        multifocus_fusion_laplacian,
        multifocus_fusion_guided,
        run_panorama_fusion,
    )
except ImportError:
    multifocus_fusion_laplacian = multifocus_fusion_guided = None
    run_panorama_fusion = None

# Light Field
try:
    from pwm_core.recon.light_field_solver import (
        shift_and_sum,
        lfbm5d,
        shift_and_sum_recon,
        lfbm5d_recon,
        lfssr_recon,
        run_light_field,
    )
except ImportError:
    shift_and_sum = lfbm5d = run_light_field = None
    shift_and_sum_recon = lfbm5d_recon = lfssr_recon = None

# Integral Photography
try:
    from pwm_core.recon.integral_solver import (
        depth_estimation,
        dibr,
        run_integral,
    )
except ImportError:
    depth_estimation = dibr = run_integral = None

# Phase Retrieval / CDI
try:
    from pwm_core.recon.phase_retrieval_solver import (
        hio,
        raar,
        gerchberg_saxton,
        run_phase_retrieval,
    )
except ImportError:
    hio = raar = gerchberg_saxton = run_phase_retrieval = None

# FLIM
try:
    from pwm_core.recon.flim_solver import (
        phasor_recon,
        mle_fit_recon,
        run_flim,
    )
except ImportError:
    phasor_recon = mle_fit_recon = run_flim = None

# Photoacoustic
try:
    from pwm_core.recon.photoacoustic_solver import (
        back_projection,
        time_reversal,
        run_photoacoustic,
    )
except ImportError:
    back_projection = time_reversal = run_photoacoustic = None

# OCT
try:
    from pwm_core.recon.oct_solver import (
        fft_recon,
        spectral_estimation,
        spectral_estimation_recon,
        oct_denoising_net_recon,
        run_oct,
    )
except ImportError:
    fft_recon = spectral_estimation = run_oct = None
    spectral_estimation_recon = oct_denoising_net_recon = None

# FPM
try:
    from pwm_core.recon.fpm_solver import (
        sequential_phase_retrieval,
        gradient_descent_fpm,
        run_fpm,
    )
except ImportError:
    sequential_phase_retrieval = gradient_descent_fpm = run_fpm = None

# DOT
try:
    from pwm_core.recon.dot_solver import (
        born_approx,
        lbfgs_tv,
        run_dot,
    )
except ImportError:
    born_approx = lbfgs_tv = run_dot = None

__all__ = [
    # Portfolio
    "run_portfolio",
    # PnP
    "run_pnp", "pnp_hqs", "pnp_admm", "pnp_fista", "get_denoiser",
    # Richardson-Lucy
    "richardson_lucy_2d", "richardson_lucy_3d",
    "richardson_lucy_operator", "run_richardson_lucy",
    # CT
    "fbp_2d", "sart_2d", "sart_operator", "run_fbp", "run_sart",
    # CS
    "tval3", "ista", "fista", "tv_prox_2d", "run_tval3", "run_ista",
    # MST (CASSI)
    "MST", "mst_recon_cassi", "shift_torch", "shift_back_meas_torch",
    # GAP-TV
    "gap_tv_cassi", "gap_tv_cacti", "gap_tv_operator",
    "run_gap_tv", "run_gap_denoise",
    # MRI
    "estimate_sensitivity_maps", "sense_reconstruction",
    "cs_mri_wavelet", "zero_filled_reconstruction",
    "run_espirit_recon", "run_cs_mri",
    # SIM
    "wiener_sim_2d", "hifi_sim_2d", "fairsim_reconstruction",
    "run_sim_reconstruction",
    # Ptychography
    "epie", "pie", "ml_epie", "create_probe", "run_epie",
    # Holography
    "angular_spectrum_propagate", "extract_plus_one_order",
    "reconstruct_offaxis_hologram", "tv_holography_admm",
    "run_holography_reconstruction",
    # CARE (widefield, confocal)
    "run_care",
    # Noise2Void
    "run_noise2void",
    # HDNet (CASSI)
    "run_hdnet",
    # HSI-SDeCNN
    "run_hsi_sdecnn",
    # EfficientSCI (CACTI)
    "run_efficientsci",
    # ELP-Unfolding (CACTI)
    "run_elp_unfolding",
    # HATNet (SPC)
    "run_hatnet",
    # ISTA-Net (SPC)
    "run_ista_net",
    # RED-CNN (CT)
    "run_redcnn",
    # VarNet (MRI)
    "run_varnet",
    # MoDL (MRI)
    "run_modl",
    # PtychoNN (ptychography)
    "run_ptychonn",
    # PhaseNet (holography)
    "run_phasenet",
    # DL-SIM
    "run_dl_sim",
    # Lensless
    "run_lensless",
    # FlatNet (lensless)
    "run_flatnet",
    # Light-sheet
    "run_lightsheet",
    # DeStripe
    "run_destripe",
    # LISTA (generic)
    "run_lista",
    # IFCNN (fusion)
    "run_ifcnn",
    # Diffusion posterior
    "run_diffusion_posterior",
    # NeRF
    "run_nerf",
    # Gaussian Splatting
    "run_gaussian_splatting",
    # Panorama/Multi-Focus
    "multifocus_fusion_laplacian", "multifocus_fusion_guided",
    "run_panorama_fusion",
    # Light Field
    "shift_and_sum", "lfbm5d", "run_light_field",
    "shift_and_sum_recon", "lfbm5d_recon", "lfssr_recon",
    # Integral Photography
    "depth_estimation", "dibr", "run_integral",
    # Phase Retrieval / CDI
    "hio", "raar", "gerchberg_saxton", "run_phase_retrieval",
    # FLIM
    "phasor_recon", "mle_fit_recon", "run_flim",
    # Photoacoustic
    "back_projection", "time_reversal", "run_photoacoustic",
    # OCT
    "fft_recon", "spectral_estimation", "run_oct",
    "spectral_estimation_recon", "oct_denoising_net_recon",
    # FPM
    "sequential_phase_retrieval", "gradient_descent_fpm", "run_fpm",
    # DOT
    "born_approx", "lbfgs_tv", "run_dot",
]
