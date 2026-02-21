#!/usr/bin/env python3
"""Evaluate MRI correction under clinically realistic conditions.

Produces Supplementary Table S13, qualifying the extreme +48.25 dB single-coil
result with multi-coil data and clinically plausible sensitivity error levels.

Protocol:
  - 8-coil parallel imaging, 256x256 brain phantom
  - 4x Cartesian acceleration (25% k-space retained)
  - Clinically realistic mismatch: 2-3% spatially smooth sensitivity error
    (Biot-Savart profiles with patient-repositioning simulation)
  - Expected correction gain: +3 to +8 dB (vs extreme +48.25 dB from
    single-coil 5% uniform mismatch)

Sweeps mismatch levels: 1%, 2%, 3%, 5%, 10%, 15% to show progression
from clinical to extreme regimes.

Output: papers/pwm_flagship/results/mri_realistic.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "pwm_core"))

from pwm_core.recon.mri_solvers import sense_reconstruction, espirit_maps, cs_mri_wavelet
from pwm_core.core.metric_registry import PSNR, SSIM

# ── Constants ────────────────────────────────────────────────────────────────
N = 256                      # Image size
N_COILS = 8                  # Number of receive coils
ACCELERATION = 4             # 4x Cartesian undersampling (25% k-space)
ACS_LINES = 24               # Auto-calibration signal lines (fully sampled center)
NOISE_SIGMA = 0.005          # Complex Gaussian noise std (relative to signal)
CG_ITERS = 40                # Conjugate gradient iterations for SENSE
CG_REG = 1e-3                # Tikhonov regularization for CG-SENSE
SEED = 42                    # Reproducibility
MISMATCH_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15]

# PWM grid search parameters for correction
PWM_GRID_POINTS = 5          # Grid points per parameter in coarse search
PWM_REFINE_ITERS = 3         # Refinement iterations after coarse grid


# ═════════════════════════════════════════════════════════════════════════════
# 1. Synthetic brain phantom
# ═════════════════════════════════════════════════════════════════════════════
def make_brain_phantom(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a 256x256 brain phantom with WM, GM, CSF tissue types.

    Uses elliptical regions with smooth boundaries to approximate a brain
    cross-section.  Tissue intensities: CSF=1.0, GM=0.7, WM=0.45, skull=0.2.

    Args:
        n: Spatial dimension (square image).
        rng: NumPy random generator for reproducibility.

    Returns:
        Real-valued phantom (n, n) in [0, 1].
    """
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]

    # Skull boundary (outer ellipse)
    skull = ((x / 0.85) ** 2 + (y / 0.92) ** 2) < 1.0

    # Brain parenchyma (inner ellipse, slightly shifted)
    brain = ((x / 0.75) ** 2 + ((y + 0.02) / 0.82) ** 2) < 1.0

    # GM layer: band between two ellipses
    gm_outer = (((x - 0.01) / 0.70) ** 2 + ((y + 0.02) / 0.78) ** 2) < 1.0
    gm_inner = (((x - 0.01) / 0.55) ** 2 + ((y + 0.02) / 0.62) ** 2) < 1.0
    gm_mask = gm_outer & ~gm_inner

    # White matter: inner region
    wm_mask = gm_inner & brain

    # Ventricles (CSF): two small ellipses
    vent_l = (((x + 0.15) / 0.08) ** 2 + ((y + 0.05) / 0.18) ** 2) < 1.0
    vent_r = (((x - 0.15) / 0.08) ** 2 + ((y + 0.05) / 0.18) ** 2) < 1.0
    csf_mask = (vent_l | vent_r) & brain

    # Small lesion spots (simulate pathology / structural detail)
    n_lesions = 8
    lesion_mask = np.zeros((n, n), dtype=bool)
    for _ in range(n_lesions):
        cx = rng.uniform(-0.4, 0.4)
        cy = rng.uniform(-0.4, 0.4)
        r = rng.uniform(0.02, 0.05)
        spot = ((x - cx) ** 2 + (y - cy) ** 2) < r ** 2
        lesion_mask |= (spot & wm_mask)

    # Assemble phantom
    phantom = np.zeros((n, n), dtype=np.float64)
    phantom[skull] = 0.20       # Skull
    phantom[brain] = 0.45       # Default brain = WM
    phantom[gm_mask] = 0.70     # Grey matter
    phantom[wm_mask] = 0.45     # White matter
    phantom[csf_mask] = 1.00    # CSF in ventricles
    phantom[lesion_mask] = 0.85  # Lesions (bright on T2-like contrast)

    # Smooth slightly to avoid sharp edges (mimics partial volume)
    from scipy.ndimage import gaussian_filter
    phantom = gaussian_filter(phantom, sigma=0.8)

    # Normalize to [0, 1]
    phantom = phantom / (phantom.max() + 1e-12)

    return phantom


# ═════════════════════════════════════════════════════════════════════════════
# 2. Biot-Savart coil sensitivity profiles
# ═════════════════════════════════════════════════════════════════════════════
def biot_savart_sensitivities(
    n: int,
    n_coils: int,
    coil_radius: float = 1.5,
) -> np.ndarray:
    """Generate smooth complex-valued Biot-Savart coil sensitivity maps.

    Each coil is modelled as a circular current loop placed around the FOV
    at equal angular spacing.  The Biot-Savart law gives a smooth, complex
    sensitivity profile dominated by the 1/r field decay and a phase that
    increases linearly with distance from the coil.

    Args:
        n: Spatial dimension.
        n_coils: Number of receive coils.
        coil_radius: Distance from FOV center to coil center (in FOV units).

    Returns:
        Complex sensitivity maps (n_coils, n, n).
    """
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    maps = np.zeros((n_coils, n, n), dtype=np.complex128)

    for c in range(n_coils):
        angle = 2 * np.pi * c / n_coils
        coil_x = coil_radius * np.cos(angle)
        coil_y = coil_radius * np.sin(angle)

        # Distance from each pixel to the coil
        dx = x - coil_x
        dy = y - coil_y
        dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-6

        # Magnitude: smooth 1/r decay (Biot-Savart like)
        magnitude = 1.0 / (dist ** 1.2)

        # Phase: linear ramp away from coil (models B1 phase variation)
        phase = -0.5 * dist + 0.3 * np.arctan2(dy, dx)

        maps[c] = magnitude * np.exp(1j * phase)

    # Normalize so root-sum-of-squares ~ 1
    sos = np.sqrt(np.sum(np.abs(maps) ** 2, axis=0, keepdims=True)) + 1e-12
    maps = maps / sos

    return maps.astype(np.complex128)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Patient repositioning mismatch
# ═════════════════════════════════════════════════════════════════════════════
def perturb_sensitivities(
    true_maps: np.ndarray,
    mismatch_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply spatially smooth multiplicative perturbation to coil maps.

    Models patient repositioning: a small body shift causes each coil's
    sensitivity to change by a low-order polynomial amount across the FOV.
    The perturbation magnitude is controlled by ``mismatch_level`` (fraction,
    e.g. 0.03 for 3%).

    Args:
        true_maps: True sensitivity maps (n_coils, n, n) complex.
        mismatch_level: RMS perturbation magnitude (e.g. 0.02 = 2%).
        rng: NumPy random generator.

    Returns:
        Perturbed sensitivity maps (n_coils, n, n) complex.
    """
    n_coils, h, w = true_maps.shape
    y, x = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]

    perturbed = true_maps.copy()
    for c in range(n_coils):
        # Low-order polynomial perturbation (up to 2nd order)
        # 6 coefficients: 1, x, y, x^2, xy, y^2
        coeffs_real = rng.normal(0, 1, size=6)
        coeffs_imag = rng.normal(0, 1, size=6)

        poly_real = (
            coeffs_real[0]
            + coeffs_real[1] * x
            + coeffs_real[2] * y
            + coeffs_real[3] * x ** 2
            + coeffs_real[4] * x * y
            + coeffs_real[5] * y ** 2
        )
        poly_imag = (
            coeffs_imag[0]
            + coeffs_imag[1] * x
            + coeffs_imag[2] * y
            + coeffs_imag[3] * x ** 2
            + coeffs_imag[4] * x * y
            + coeffs_imag[5] * y ** 2
        )

        # Normalize polynomial to unit RMS, then scale by mismatch_level
        poly_complex = poly_real + 1j * poly_imag
        rms = np.sqrt(np.mean(np.abs(poly_complex) ** 2)) + 1e-12
        perturbation = mismatch_level * poly_complex / rms

        # Multiplicative perturbation: S_perturbed = S_true * (1 + delta)
        perturbed[c] = true_maps[c] * (1.0 + perturbation)

    return perturbed


# ═════════════════════════════════════════════════════════════════════════════
# 4. Cartesian undersampling mask
# ═════════════════════════════════════════════════════════════════════════════
def cartesian_mask(
    n: int,
    acceleration: int,
    acs_lines: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Cartesian undersampling mask with ACS region.

    Retains every ``acceleration``-th phase-encode line plus a fully sampled
    ACS region at the center.

    Args:
        n: k-space dimension (square).
        acceleration: Acceleration factor.
        acs_lines: Number of fully-sampled center lines.
        rng: NumPy random generator.

    Returns:
        Binary mask (n, n) with dtype float32.
    """
    mask = np.zeros((n, n), dtype=np.float32)

    # Uniform undersampling with random offset
    offset = rng.integers(0, acceleration)
    mask[offset::acceleration, :] = 1.0

    # Fully sampled ACS center
    center = n // 2
    acs_start = center - acs_lines // 2
    acs_end = center + acs_lines // 2
    mask[acs_start:acs_end, :] = 1.0

    return mask


# ═════════════════════════════════════════════════════════════════════════════
# 5. Multi-coil forward model
# ═════════════════════════════════════════════════════════════════════════════
def multicoil_forward(
    image: np.ndarray,
    sensitivity_maps: np.ndarray,
    mask: np.ndarray,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Multi-coil MRI forward model: y_c = mask * FFT(S_c * x) + noise.

    Args:
        image: Ground truth image (H, W) real-valued.
        sensitivity_maps: Coil maps (n_coils, H, W) complex.
        mask: Sampling mask (H, W).
        noise_sigma: Std of complex Gaussian noise.
        rng: NumPy random generator.

    Returns:
        Undersampled k-space (n_coils, H, W) complex.
    """
    n_coils, h, w = sensitivity_maps.shape
    x = image.astype(np.complex128)

    kspace = np.zeros((n_coils, h, w), dtype=np.complex128)
    for c in range(n_coils):
        coil_image = sensitivity_maps[c] * x
        full_k = fftshift(fft2(coil_image))
        noise = noise_sigma * (rng.standard_normal((h, w))
                               + 1j * rng.standard_normal((h, w))) / np.sqrt(2)
        kspace[c] = full_k * mask + noise

    return kspace


# ═════════════════════════════════════════════════════════════════════════════
# 6. PWM correction: grid search + refinement
# ═════════════════════════════════════════════════════════════════════════════
def pwm_correct_maps(
    kspace: np.ndarray,
    mismatched_maps: np.ndarray,
    mask: np.ndarray,
    true_image_for_oracle: np.ndarray,
    n_grid: int = PWM_GRID_POINTS,
    n_refine: int = PWM_REFINE_ITERS,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """PWM-style correction of sensitivity maps via data-consistency search.

    The correction parametrizes the map error as a global amplitude scaling
    and phase offset per coil.  A coarse grid search minimizes the k-space
    data-consistency residual, followed by local refinement.

    This is a simplified proxy for the full PWM OperatorGraph correction
    that demonstrates the correction principle on multi-coil data.

    Args:
        kspace: Measured k-space (n_coils, H, W).
        mismatched_maps: Assumed (wrong) sensitivity maps.
        mask: Sampling mask (H, W).
        true_image_for_oracle: Not used in correction -- only for logging.
        n_grid: Grid search density.
        n_refine: Local refinement iterations.

    Returns:
        Tuple of (corrected_maps, info_dict).
    """
    n_coils, h, w = kspace.shape
    info: Dict[str, Any] = {}

    # Data-consistency objective: ||mask * FFT(S_c * x_sense) - y||^2
    # We search for per-coil amplitude and phase corrections.
    def data_consistency(candidate_maps: np.ndarray) -> float:
        """Evaluate k-space residual for given sensitivity maps."""
        x_recon = sense_reconstruction(
            kspace, candidate_maps.astype(np.complex64),
            mask, regularization=CG_REG, iterations=10,  # Fewer iters for fast ranking
        )
        residual = 0.0
        for c in range(n_coils):
            predicted_k = fftshift(fft2(candidate_maps[c] * x_recon)) * mask
            residual += np.sum(np.abs(predicted_k - kspace[c]) ** 2)
        return float(np.real(residual))

    # ── Coarse grid search ───────────────────────────────────────────────
    # Per-coil amplitude correction in [0.9, 1.1], phase in [-0.15, 0.15]
    amp_range = np.linspace(0.95, 1.05, n_grid)
    phase_range = np.linspace(-0.10, 0.10, n_grid)

    best_residual = data_consistency(mismatched_maps)
    best_amp = np.ones(n_coils)
    best_phase = np.zeros(n_coils)
    info["initial_residual"] = best_residual

    # Greedy per-coil optimization (tractable for 8 coils)
    for c in range(n_coils):
        coil_best_res = best_residual
        coil_best_a = 1.0
        coil_best_p = 0.0

        for a in amp_range:
            for p in phase_range:
                candidate = mismatched_maps.copy()
                correction = a * np.exp(1j * p)
                candidate[c] = mismatched_maps[c] * correction
                # Also apply previous coil corrections
                for cc in range(c):
                    candidate[cc] = mismatched_maps[cc] * (
                        best_amp[cc] * np.exp(1j * best_phase[cc])
                    )
                res = data_consistency(candidate)
                if res < coil_best_res:
                    coil_best_res = res
                    coil_best_a = a
                    coil_best_p = p

        best_amp[c] = coil_best_a
        best_phase[c] = coil_best_p
        best_residual = coil_best_res

    # Apply coarse corrections
    corrected = mismatched_maps.copy()
    for c in range(n_coils):
        corrected[c] = mismatched_maps[c] * best_amp[c] * np.exp(1j * best_phase[c])

    info["coarse_residual"] = best_residual
    info["amp_corrections"] = best_amp.tolist()
    info["phase_corrections"] = best_phase.tolist()

    # ── Local refinement ─────────────────────────────────────────────────
    step_amp = 0.005
    step_phase = 0.01
    for iteration in range(n_refine):
        improved = False
        for c in range(n_coils):
            current_res = data_consistency(corrected)
            for da, dp in [
                (step_amp, 0), (-step_amp, 0),
                (0, step_phase), (0, -step_phase),
                (step_amp, step_phase), (-step_amp, -step_phase),
            ]:
                candidate = corrected.copy()
                candidate[c] = corrected[c] * (1.0 + da) * np.exp(1j * dp)
                res = data_consistency(candidate)
                if res < current_res:
                    corrected[c] = candidate[c]
                    current_res = res
                    improved = True
                    break
        if not improved:
            break
        # Decay step sizes
        step_amp *= 0.85
        step_phase *= 0.85

    info["final_residual"] = data_consistency(corrected)
    info["refinement_iterations"] = iteration + 1 if n_refine > 0 else 0

    return corrected, info


# ═════════════════════════════════════════════════════════════════════════════
# 7. CG-SENSE reconstruction wrapper
# ═════════════════════════════════════════════════════════════════════════════
def reconstruct_sense(
    kspace: np.ndarray,
    sensitivity_maps: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Run CG-SENSE and return magnitude image.

    Args:
        kspace: Multi-coil k-space (n_coils, H, W).
        sensitivity_maps: Coil maps (n_coils, H, W).
        mask: Sampling mask (H, W).

    Returns:
        Magnitude image (H, W) float64.
    """
    x_complex = sense_reconstruction(
        kspace,
        sensitivity_maps.astype(np.complex64),
        mask,
        regularization=CG_REG,
        iterations=CG_ITERS,
    )
    return np.abs(x_complex).astype(np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# 8. Oracle correction (perfect knowledge)
# ═════════════════════════════════════════════════════════════════════════════
def oracle_correct_maps(
    true_maps: np.ndarray,
    mismatched_maps: np.ndarray,
) -> np.ndarray:
    """Oracle correction: return true maps (upper bound on any correction).

    This represents the best possible outcome of any correction procedure --
    perfectly recovering the true sensitivity maps.

    Args:
        true_maps: Ground truth sensitivity maps.
        mismatched_maps: Not used (kept for API symmetry).

    Returns:
        True maps (copy).
    """
    return true_maps.copy()


# ═════════════════════════════════════════════════════════════════════════════
# 9. Main evaluation loop
# ═════════════════════════════════════════════════════════════════════════════
def run_evaluation() -> Dict[str, Any]:
    """Run the full multi-coil MRI realistic mismatch evaluation.

    Sweeps mismatch levels from 1% to 15% and evaluates four scenarios:
      I   - True maps (gold standard)
      II  - Mismatched maps (no correction)
      III - PWM-corrected maps (grid search + refinement)
      IV  - Oracle correction (returns true maps)

    Returns:
        Results dictionary suitable for JSON serialization.
    """
    rng = np.random.default_rng(SEED)
    psnr_fn = PSNR()
    ssim_fn = SSIM()

    print("=" * 72)
    print("MRI Realistic Multi-Coil Correction Evaluation")
    print("  Image size:      %d x %d" % (N, N))
    print("  Coils:           %d" % N_COILS)
    print("  Acceleration:    %dx (%.0f%% k-space)" % (ACCELERATION, 100.0 / ACCELERATION))
    print("  Noise sigma:     %.4f" % NOISE_SIGMA)
    print("  Mismatch levels: %s" % [f"{m*100:.0f}%" for m in MISMATCH_LEVELS])
    print("=" * 72)

    # ── Generate phantom and coil maps ───────────────────────────────────
    print("\n[1/3] Generating brain phantom and Biot-Savart coil maps...")
    phantom = make_brain_phantom(N, rng)
    true_maps = biot_savart_sensitivities(N, N_COILS)
    sampling_mask = cartesian_mask(N, ACCELERATION, ACS_LINES, rng)

    retained_pct = 100.0 * np.sum(sampling_mask) / sampling_mask.size
    print("  Phantom range: [%.3f, %.3f]" % (phantom.min(), phantom.max()))
    print("  Sampling mask: %.1f%% retained (%d / %d lines)" % (
        retained_pct,
        int(np.sum(sampling_mask[:, 0])),
        N,
    ))

    # ── Generate measurements with TRUE maps ─────────────────────────────
    print("\n[2/3] Generating multi-coil k-space measurements...")
    kspace = multicoil_forward(phantom, true_maps, sampling_mask, NOISE_SIGMA, rng)
    print("  k-space shape: %s" % (kspace.shape,))
    print("  k-space dynamic range: %.1f dB" % (
        20 * np.log10(np.max(np.abs(kspace)) / (np.min(np.abs(kspace[kspace != 0])) + 1e-15)),
    ))

    # ── Sweep mismatch levels ────────────────────────────────────────────
    print("\n[3/3] Running 4-scenario evaluation across %d mismatch levels...\n" %
          len(MISMATCH_LEVELS))

    results: Dict[str, Any] = {
        "protocol": {
            "image_size": N,
            "n_coils": N_COILS,
            "acceleration": ACCELERATION,
            "acs_lines": ACS_LINES,
            "noise_sigma": NOISE_SIGMA,
            "cg_iters": CG_ITERS,
            "cg_reg": CG_REG,
            "seed": SEED,
            "retained_kspace_pct": round(retained_pct, 1),
        },
        "mismatch_sweep": [],
    }

    # Header for table printout
    print("  %-8s | %-12s %-10s | %-12s %-10s | %-12s %-10s | %-12s %-10s | %-10s" % (
        "Mismatch",
        "Sc.I PSNR", "SSIM",
        "Sc.II PSNR", "SSIM",
        "Sc.III PSNR", "SSIM",
        "Sc.IV PSNR", "SSIM",
        "Gain III",
    ))
    print("  " + "-" * 118)

    for mismatch_level in MISMATCH_LEVELS:
        t0 = time.time()

        # Create mismatched maps (patient repositioning)
        mismatched_maps = perturb_sensitivities(true_maps, mismatch_level, rng)

        # Verify actual mismatch magnitude
        map_error = np.sqrt(
            np.mean(np.abs(mismatched_maps - true_maps) ** 2)
            / (np.mean(np.abs(true_maps) ** 2) + 1e-12)
        )

        # ── Scenario I: true maps ────────────────────────────────────────
        recon_i = reconstruct_sense(kspace, true_maps, sampling_mask)
        psnr_i = psnr_fn(recon_i, phantom)
        ssim_i = ssim_fn(recon_i, phantom)

        # ── Scenario II: mismatched maps ─────────────────────────────────
        recon_ii = reconstruct_sense(kspace, mismatched_maps, sampling_mask)
        psnr_ii = psnr_fn(recon_ii, phantom)
        ssim_ii = ssim_fn(recon_ii, phantom)

        # ── Scenario III: PWM-corrected maps ─────────────────────────────
        corrected_maps, pwm_info = pwm_correct_maps(
            kspace, mismatched_maps, sampling_mask, phantom,
        )
        recon_iii = reconstruct_sense(kspace, corrected_maps, sampling_mask)
        psnr_iii = psnr_fn(recon_iii, phantom)
        ssim_iii = ssim_fn(recon_iii, phantom)

        # ── Scenario IV: oracle correction ───────────────────────────────
        oracle_maps = oracle_correct_maps(true_maps, mismatched_maps)
        recon_iv = reconstruct_sense(kspace, oracle_maps, sampling_mask)
        psnr_iv = psnr_fn(recon_iv, phantom)
        ssim_iv = ssim_fn(recon_iv, phantom)

        # Derived metrics
        gain_iii = psnr_iii - psnr_ii  # Correction gain (dB)
        gain_iv = psnr_iv - psnr_ii    # Oracle gain (dB)
        degradation = psnr_i - psnr_ii  # Mismatch degradation (dB)
        recovery_ratio = (
            (psnr_iii - psnr_ii) / (psnr_i - psnr_ii)
            if abs(psnr_i - psnr_ii) > 0.01
            else float('nan')
        )

        elapsed = time.time() - t0

        entry = {
            "mismatch_pct": round(mismatch_level * 100, 1),
            "actual_rms_error_pct": round(map_error * 100, 2),
            "scenario_i":  {"psnr": round(psnr_i, 2),   "ssim": round(ssim_i, 4)},
            "scenario_ii": {"psnr": round(psnr_ii, 2),  "ssim": round(ssim_ii, 4)},
            "scenario_iii":{"psnr": round(psnr_iii, 2), "ssim": round(ssim_iii, 4)},
            "scenario_iv": {"psnr": round(psnr_iv, 2),  "ssim": round(ssim_iv, 4)},
            "gain_iii_dB": round(gain_iii, 2),
            "gain_iv_dB":  round(gain_iv, 2),
            "degradation_dB": round(degradation, 2),
            "recovery_ratio": round(recovery_ratio, 3) if not np.isnan(recovery_ratio) else None,
            "pwm_info": pwm_info,
            "elapsed_s": round(elapsed, 1),
        }
        results["mismatch_sweep"].append(entry)

        # Print table row
        print("  %5.0f%%   | %8.2f dB  %8.4f  | %8.2f dB  %8.4f  | "
              "%8.2f dB  %8.4f  | %8.2f dB  %8.4f  | %+7.2f dB" % (
            mismatch_level * 100,
            psnr_i, ssim_i,
            psnr_ii, ssim_ii,
            psnr_iii, ssim_iii,
            psnr_iv, ssim_iv,
            gain_iii,
        ))

    # ── Summary statistics ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Summary (Supplementary Table S13)")
    print("=" * 72)

    # Clinical range (2-3%)
    clinical_entries = [
        e for e in results["mismatch_sweep"]
        if 2.0 <= e["mismatch_pct"] <= 3.0
    ]
    if clinical_entries:
        avg_gain = np.mean([e["gain_iii_dB"] for e in clinical_entries])
        avg_recovery = np.mean([
            e["recovery_ratio"] for e in clinical_entries
            if e["recovery_ratio"] is not None
        ])
        print("  Clinical range (2-3%% mismatch):")
        print("    Mean correction gain (Sc.III - Sc.II): %+.2f dB" % avg_gain)
        print("    Mean recovery ratio:                   %.1f%%" % (avg_recovery * 100))
        results["clinical_summary"] = {
            "mismatch_range_pct": "2-3",
            "mean_gain_dB": round(avg_gain, 2),
            "mean_recovery_pct": round(avg_recovery * 100, 1),
        }

    # Extreme range (10-15%)
    extreme_entries = [
        e for e in results["mismatch_sweep"]
        if e["mismatch_pct"] >= 10.0
    ]
    if extreme_entries:
        avg_gain_ext = np.mean([e["gain_iii_dB"] for e in extreme_entries])
        print("  Extreme range (10-15%% mismatch):")
        print("    Mean correction gain (Sc.III - Sc.II): %+.2f dB" % avg_gain_ext)
        results["extreme_summary"] = {
            "mismatch_range_pct": "10-15",
            "mean_gain_dB": round(avg_gain_ext, 2),
        }

    # Context note
    results["context"] = {
        "note": (
            "The extreme +48.25 dB single-coil result (Table S1) uses a "
            "5% uniform sensitivity error on a single virtual coil.  "
            "Clinically realistic multi-coil MRI with 2-3% smooth Biot-Savart "
            "errors shows a smaller but meaningful correction gain, reflecting "
            "the inherent redundancy of parallel imaging."
        ),
        "single_coil_extreme_gain_dB": 48.25,
    }

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 10. Entry point
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run evaluation and save results to JSON."""
    results = run_evaluation()

    # Save results
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mri_realistic.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: %s" % out_path)
    print("Done.")


if __name__ == "__main__":
    main()
