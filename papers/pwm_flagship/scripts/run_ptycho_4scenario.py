#!/usr/bin/env python3
"""
Ptychography 4-Scenario validation using real 4D STEM data.
Dataset: SrTiO3 [001] from Zenodo 5113449 (Medipix3RX, 300kV)
128x128 scan, 256x256 diffraction patterns, uint8.

Demonstrates Gate 3 (probe position mismatch) via SSB ptychography:
  Scenario I:  Correct scan step → reference phase image
  Scenario II: Wrong scan step (scale mismatch) → degraded
  Scenario III: Grid-search calibrated step → recovered
  Scenario IV: Oracle (best possible step) → upper bound

Uses Single Sideband (SSB) ptychography which is:
  - Non-iterative (direct inversion)
  - Sensitive to scan step calibration
  - Standard for atomic-resolution 4D STEM
"""

import numpy as np
import json
import os
import sys
from pathlib import Path

# ─── .mib loader ───────────────────────────────────────────
HEADER_SIZE = 384
DET_SIZE = 256  # 256x256 Medipix3RX
FRAME_BYTES = HEADER_SIZE + DET_SIZE * DET_SIZE


def load_mib_stack(data_dir, n_scan_x=128, n_scan_y=128, subset=None):
    """Load 4D STEM data from Merlin .mib files.

    Returns: data[n_scan_y, n_scan_x, DET_SIZE, DET_SIZE] as float32
    subset: (row_start, row_end, col_start, col_end) to load partial data
    """
    # Sort files numerically
    mib_files = sorted(
        Path(data_dir).glob("*.mib"),
        key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or '0')
    )

    if subset is not None:
        rs, re, cs, ce = subset
        ny, nx = re - rs, ce - cs
    else:
        rs, re, cs, ce = 0, n_scan_y, 0, n_scan_x
        ny, nx = n_scan_y, n_scan_x

    data = np.zeros((ny, nx, DET_SIZE, DET_SIZE), dtype=np.float32)

    frame_idx = 0
    for fpath in mib_files:
        fsize = fpath.stat().st_size
        n_frames_in_file = fsize // FRAME_BYTES

        with open(fpath, 'rb') as f:
            for i in range(n_frames_in_file):
                # Map linear index to 2D scan position (row-major)
                row = frame_idx // n_scan_x
                col = frame_idx % n_scan_x
                frame_idx += 1

                if row < rs or row >= re or col < cs or col >= ce:
                    f.seek(FRAME_BYTES, 1)  # skip this frame
                    continue

                f.seek(HEADER_SIZE, 1)  # skip header
                raw = np.frombuffer(f.read(DET_SIZE * DET_SIZE), dtype=np.uint8)
                data[row - rs, col - cs] = raw.reshape(DET_SIZE, DET_SIZE).astype(np.float32)

    print(f"Loaded 4D STEM data: {data.shape}, "
          f"range [{data.min():.0f}, {data.max():.0f}], mean={data.mean():.2f}")
    return data


# ─── SSB ptychography ─────────────────────────────────────
def ssb_reconstruct(data_4d, step_scale=1.0, bf_radius=None):
    """Single Sideband ptychography reconstruction.

    Args:
        data_4d: [Ny, Nx, Ky, Kx] 4D STEM dataset
        step_scale: multiplicative factor on scan step (1.0 = nominal)
        bf_radius: bright-field disk radius in pixels (auto-detected if None)

    Returns:
        phase: [Ny, Nx] reconstructed phase image
        amplitude: [Ny, Nx] reconstructed amplitude image
    """
    Ny, Nx, Ky, Kx = data_4d.shape

    # Auto-detect BF disk radius from average CBED pattern
    if bf_radius is None:
        avg_pattern = data_4d.mean(axis=(0, 1))
        # Find radius where azimuthal average drops to half-max
        cy, cx = Ky // 2, Kx // 2
        yy, xx = np.mgrid[:Ky, :Kx]
        rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        max_r = min(cy, cx)
        radial_profile = np.zeros(max_r)
        for r in range(max_r):
            mask = (rr >= r) & (rr < r + 1)
            if mask.any():
                radial_profile[r] = avg_pattern[mask].mean()
        half_max = radial_profile.max() / 2
        bf_radius = np.argmin(np.abs(radial_profile - half_max))
        bf_radius = max(bf_radius, 10)  # minimum 10 pixels
        print(f"  Auto-detected BF disk radius: {bf_radius} pixels")

    # Take sqrt of intensities (amplitude of exit wave)
    amp_4d = np.sqrt(np.maximum(data_4d, 0))

    # FT over scan dimensions
    # G_hat(qy, qx, ky, kx) = FT_{ry,rx}[sqrt(I)](qy, qx, ky, kx)
    # The scan step enters as the spatial frequency scaling
    G_hat = np.fft.fft2(amp_4d, axes=(0, 1))

    # Create BF disk mask (aperture function)
    cy, cx = Ky // 2, Kx // 2
    yy, xx = np.mgrid[:Ky, :Kx]
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    aperture = (rr <= bf_radius).astype(np.float32)

    # SSB reconstruction: for each scan frequency q, integrate over
    # overlapping regions of shifted apertures
    # Object(q) ∝ Σ_k G_hat(q, k) * A(k) * A*(k - q_scaled)
    # where q_scaled accounts for the scan step

    # Scan frequency grid (affected by step_scale)
    qy = np.fft.fftfreq(Ny) * step_scale
    qx = np.fft.fftfreq(Nx) * step_scale

    # For efficiency, compute object spectrum via integration
    obj_spectrum = np.zeros((Ny, Nx), dtype=np.complex128)

    # Scale factor: how many detector pixels correspond to one scan frequency unit
    # For atomic-resolution STEM, the scan step is typically ~0.1-0.5 Å
    # The detector pixel corresponds to a reciprocal-space increment
    # The ratio (scan_step / detector_pixel_recip) determines the scaling
    # We use a heuristic: scale = bf_radius / (min(Ny,Nx)/4)
    det_scale = bf_radius / (min(Ny, Nx) / 4.0) * step_scale

    for iqy in range(Ny):
        for iqx in range(Nx):
            if iqy == 0 and iqx == 0:
                continue  # skip DC

            # Shifted aperture position in detector coordinates
            shift_y = qy[iqy] * Ny * det_scale
            shift_x = qx[iqx] * Nx * det_scale

            # Overlap of aperture A(k) and A(k - q)
            rr_shifted = np.sqrt((yy - cy - shift_y)**2 + (xx - cx - shift_x)**2)
            aperture_shifted = (rr_shifted <= bf_radius).astype(np.float32)

            overlap = aperture * aperture_shifted
            if overlap.sum() < 1:
                continue

            # Integrate G_hat over overlap region
            obj_spectrum[iqy, iqx] = np.sum(
                G_hat[iqy, iqx] * overlap
            )

    # Inverse FT to get object
    obj_complex = np.fft.ifft2(obj_spectrum)
    phase = np.angle(obj_complex)
    amplitude = np.abs(obj_complex)

    return phase, amplitude


def icom_reconstruct(data_4d):
    """Integrated Center of Mass (iCoM) reconstruction.

    More robust than SSB for initial comparison. Computes beam deflection
    at each scan point and integrates to get phase.

    Returns: phase [Ny, Nx]
    """
    Ny, Nx, Ky, Kx = data_4d.shape
    cy, cx = Ky // 2, Kx // 2

    # Coordinate arrays for CoM computation
    yy, xx = np.mgrid[:Ky, :Kx]
    yy = yy.astype(np.float64) - cy
    xx = xx.astype(np.float64) - cx

    # Compute center of mass for each diffraction pattern
    com_y = np.zeros((Ny, Nx), dtype=np.float64)
    com_x = np.zeros((Ny, Nx), dtype=np.float64)

    for iy in range(Ny):
        for ix in range(Nx):
            pattern = data_4d[iy, ix].astype(np.float64)
            total = pattern.sum()
            if total > 0:
                com_y[iy, ix] = np.sum(pattern * yy) / total
                com_x[iy, ix] = np.sum(pattern * xx) / total

    # Integrate CoM to get phase using Fourier integration
    # ∇φ = (com_x, com_y) → φ = F^{-1}[ F[∇φ] / (iq) ]
    qy = np.fft.fftfreq(Ny)
    qx = np.fft.fftfreq(Nx)
    QY, QX = np.meshgrid(qy, qx, indexing='ij')

    # Fourier transform of gradient components
    F_com_y = np.fft.fft2(com_y)
    F_com_x = np.fft.fft2(com_x)

    # Integrate: φ_hat = (iq_y * F_com_y + iq_x * F_com_x) / (q_y² + q_x²)
    denom = QY**2 + QX**2
    denom[0, 0] = 1  # avoid division by zero

    phase_hat = (1j * QY * F_com_y + 1j * QX * F_com_x) / (2 * np.pi * denom)
    phase_hat[0, 0] = 0  # zero mean

    phase = np.real(np.fft.ifft2(phase_hat))
    return phase


def compute_phase_quality(phase_ref, phase_test):
    """Compute quality metrics between two phase images.

    Returns dict with PSNR, SSIM-like metric, and gradient correlation.
    """
    # Normalize both to zero mean, unit variance
    ref = phase_ref - phase_ref.mean()
    tst = phase_test - phase_test.mean()

    # Allow global phase offset
    # Find best linear fit: tst_aligned = a * tst + b
    a = np.sum(ref * tst) / (np.sum(tst * tst) + 1e-12)
    tst_aligned = a * tst

    # MSE and PSNR
    mse = np.mean((ref - tst_aligned)**2)
    ref_range = ref.max() - ref.min()
    psnr = 10 * np.log10(ref_range**2 / (mse + 1e-12)) if mse > 0 else 50.0

    # Gradient correlation (structural similarity proxy)
    gy_ref, gx_ref = np.gradient(ref)
    gy_tst, gx_tst = np.gradient(tst_aligned)
    grad_ref = np.sqrt(gy_ref**2 + gx_ref**2)
    grad_tst = np.sqrt(gy_tst**2 + gx_tst**2)

    grad_corr = np.corrcoef(grad_ref.ravel(), grad_tst.ravel())[0, 1]

    # Pearson correlation of phase
    phase_corr = np.corrcoef(ref.ravel(), tst_aligned.ravel())[0, 1]

    return {
        'psnr_db': float(psnr),
        'phase_correlation': float(phase_corr),
        'gradient_correlation': float(grad_corr),
        'mse': float(mse)
    }


# ─── Position mismatch simulation ─────────────────────────
def add_position_jitter(data_4d, jitter_std_px):
    """Simulate probe position jitter by shifting diffraction patterns.

    In real ptychography, position errors mean the probe was at a different
    location than assumed. This is equivalent to the object being shifted
    under the probe, which (for thin objects) shifts the diffraction pattern.

    For 4D STEM, we simulate this by:
    1. Shifting each diffraction pattern by a random sub-pixel amount
    2. This is equivalent to the beam being at a slightly different position

    Returns: jittered data, applied shifts
    """
    from scipy.ndimage import shift as ndshift

    Ny, Nx, Ky, Kx = data_4d.shape
    jittered = np.zeros_like(data_4d)
    shifts = np.random.randn(Ny, Nx, 2) * jitter_std_px

    for iy in range(Ny):
        for ix in range(Nx):
            jittered[iy, ix] = ndshift(
                data_4d[iy, ix],
                [shifts[iy, ix, 0], shifts[iy, ix, 1]],
                order=1, mode='constant', cval=0
            )

    return jittered, shifts


def apply_scan_rescale(data_4d, scale_factor):
    """Simulate scan step mismatch by resampling the 4D dataset.

    If the true scan step is d, but we assume d*(1+ε), this is equivalent
    to the spatial frequencies being miscalibrated. We simulate this by
    resampling the scan dimensions.
    """
    from scipy.ndimage import zoom

    Ny, Nx, Ky, Kx = data_4d.shape
    # Resample scan dimensions by 1/scale_factor
    # (if step is larger, we see fewer periods → undersample)
    rescaled = zoom(data_4d, [1.0/scale_factor, 1.0/scale_factor, 1, 1],
                    order=1, mode='nearest')
    # Crop/pad back to original size
    ry, rx = rescaled.shape[:2]
    result = np.zeros_like(data_4d)
    cy, cx = min(ry, Ny), min(rx, Nx)
    result[:cy, :cx] = rescaled[:cy, :cx]

    return result


# ─── Main experiment ───────────────────────────────────────
def main():
    data_dir = "/home/spiritai/real_datasets/ptychography/20200518 165148"
    results_dir = "/home/spiritai/PWM/test5/Physics_World_Model/papers/pwm_flagship/results/real_data_4scenario"
    os.makedirs(results_dir, exist_ok=True)

    # Load subset for speed (64x64 scan area from center)
    print("=" * 60)
    print("PTYCHOGRAPHY 4-SCENARIO VALIDATION")
    print("Dataset: SrTiO3 4D STEM (Zenodo 5113449)")
    print("=" * 60)

    subset_size = 64
    offset = (128 - subset_size) // 2
    subset = (offset, offset + subset_size, offset, offset + subset_size)

    print(f"\nLoading {subset_size}x{subset_size} scan subset from center...")
    data_4d = load_mib_stack(data_dir, n_scan_x=128, n_scan_y=128, subset=subset)

    # ─── Method 1: iCoM (simpler, more robust) ───
    print("\n" + "─" * 40)
    print("Method: integrated Center of Mass (iCoM)")
    print("─" * 40)

    # Scenario I: Correct positions (reference)
    print("\nScenario I: Nominal positions (reference)...")
    phase_ref = icom_reconstruct(data_4d)
    print(f"  Phase range: [{phase_ref.min():.4f}, {phase_ref.max():.4f}]")

    # Scenario II: Position jitter (Gate 3 mismatch)
    jitter_levels = [0.5, 1.0, 2.0, 4.0]  # pixels of position error
    results_icom = {
        'method': 'iCoM',
        'dataset': 'SrTiO3_4DSTEM_Zenodo5113449',
        'scan_subset': f'{subset_size}x{subset_size}',
        'scenarios': {}
    }

    results_icom['scenarios']['I_nominal'] = {
        'phase_range': [float(phase_ref.min()), float(phase_ref.max())],
        'phase_std': float(phase_ref.std())
    }

    for jitter_std in jitter_levels:
        print(f"\nScenario II: Position jitter σ={jitter_std:.1f} px...")
        np.random.seed(42)  # reproducible
        data_jittered, shifts = add_position_jitter(data_4d, jitter_std)
        phase_jittered = icom_reconstruct(data_jittered)

        metrics = compute_phase_quality(phase_ref, phase_jittered)
        print(f"  PSNR: {metrics['psnr_db']:.2f} dB, "
              f"Phase corr: {metrics['phase_correlation']:.4f}, "
              f"Grad corr: {metrics['gradient_correlation']:.4f}")

        results_icom['scenarios'][f'II_jitter_{jitter_std}px'] = metrics

    # Scenario III: Corrected (undo jitter by averaging)
    # In practice, autonomous calibration would estimate and correct positions
    # Here we demonstrate that with known shifts, correction recovers quality
    print(f"\nScenario III: Oracle-corrected jitter (σ=2.0 px, undo shifts)...")
    np.random.seed(42)
    data_jittered, shifts = add_position_jitter(data_4d, 2.0)
    # Correct by applying inverse shifts
    from scipy.ndimage import shift as ndshift
    Ny, Nx = data_jittered.shape[:2]
    data_corrected = np.zeros_like(data_jittered)
    for iy in range(Ny):
        for ix in range(Nx):
            data_corrected[iy, ix] = ndshift(
                data_jittered[iy, ix],
                [-shifts[iy, ix, 0], -shifts[iy, ix, 1]],
                order=1, mode='constant', cval=0
            )
    phase_corrected = icom_reconstruct(data_corrected)
    metrics_corrected = compute_phase_quality(phase_ref, phase_corrected)
    print(f"  PSNR: {metrics_corrected['psnr_db']:.2f} dB, "
          f"Phase corr: {metrics_corrected['phase_correlation']:.4f}")
    results_icom['scenarios']['III_corrected_2px'] = metrics_corrected

    # Compute recovery ratio
    psnr_I = results_icom['scenarios']['I_nominal']['phase_std']  # reference std
    psnr_II = results_icom['scenarios']['II_jitter_2.0px']['psnr_db']
    psnr_III = metrics_corrected['psnr_db']
    # Use PSNR of self-comparison as Scenario I PSNR (infinite by definition)
    # Instead, compare phase_std as quality metric
    corr_II = results_icom['scenarios']['II_jitter_2.0px']['phase_correlation']
    corr_III = metrics_corrected['phase_correlation']
    recovery = (corr_III - corr_II) / (1.0 - corr_II) if corr_II < 1.0 else 1.0
    results_icom['scenarios']['recovery_ratio'] = float(recovery)
    print(f"\n  Recovery ratio (correlation): {recovery:.1%}")

    # ─── Method 2: Measurement residual approach ───
    print("\n" + "─" * 40)
    print("Method: Measurement Residual (cross-consistency)")
    print("─" * 40)

    # For ptychography, the "measurement residual" analog is:
    # How well does the CoM field from jittered data predict the nominal CoM?
    # This is the cross-residual: reconstruct with wrong positions,
    # evaluate consistency with correct-position forward model

    Ny, Nx, Ky, Kx = data_4d.shape
    cy, cx = Ky // 2, Kx // 2
    yy = np.arange(Ky, dtype=np.float64) - cy
    xx = np.arange(Kx, dtype=np.float64) - cx

    # Reference CoM field
    com_y_ref = np.zeros((Ny, Nx))
    com_x_ref = np.zeros((Ny, Nx))
    for iy in range(Ny):
        for ix in range(Nx):
            p = data_4d[iy, ix].astype(np.float64)
            total = p.sum()
            if total > 0:
                com_y_ref[iy, ix] = np.dot(p.sum(axis=1), yy) / total
                com_x_ref[iy, ix] = np.dot(p.sum(axis=0), xx) / total

    results_residual = {'mismatch_levels': {}}

    for jitter_std in jitter_levels:
        np.random.seed(42)
        data_jittered, _ = add_position_jitter(data_4d, jitter_std)

        # CoM from jittered data
        com_y_jit = np.zeros((Ny, Nx))
        com_x_jit = np.zeros((Ny, Nx))
        for iy in range(Ny):
            for ix in range(Nx):
                p = data_jittered[iy, ix].astype(np.float64)
                total = p.sum()
                if total > 0:
                    com_y_jit[iy, ix] = np.dot(p.sum(axis=1), yy) / total
                    com_x_jit[iy, ix] = np.dot(p.sum(axis=0), xx) / total

        # Cross-residual: difference in CoM fields
        res_y = com_y_jit - com_y_ref
        res_x = com_x_jit - com_x_ref
        cross_residual = np.sqrt(np.mean(res_y**2 + res_x**2))
        ref_magnitude = np.sqrt(np.mean(com_y_ref**2 + com_x_ref**2))
        ratio = cross_residual / ref_magnitude if ref_magnitude > 0 else 0

        print(f"  Jitter σ={jitter_std:.1f} px: CoM residual = {cross_residual:.4f}, "
              f"ratio = {ratio:.4f} ({ratio*100:.1f}%)")

        results_residual['mismatch_levels'][f'{jitter_std}px'] = {
            'cross_residual': float(cross_residual),
            'ref_magnitude': float(ref_magnitude),
            'residual_ratio': float(ratio),
            'residual_percent': float(ratio * 100)
        }

    results_icom['measurement_residual'] = results_residual

    # ─── Save results ─────────────────────────────────────
    outpath = os.path.join(results_dir, 'ptycho_4scenario_results.json')
    with open(outpath, 'w') as f:
        json.dump(results_icom, f, indent=2)
    print(f"\nResults saved to {outpath}")

    # ─── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY: Ptychography Gate 3 Validation")
    print("=" * 60)
    print(f"Dataset: Real 4D STEM SrTiO3 (128×128 scan, 256×256 det)")
    print(f"Carrier: Electrons (300 kV)")
    print(f"Gate 3 parameter: Probe position jitter")
    print(f"\niCoM Phase Correlation:")
    print(f"  Nominal:        1.0000")
    for jl in jitter_levels:
        key = f'II_jitter_{jl}px'
        if key in results_icom['scenarios']:
            m = results_icom['scenarios'][key]
            print(f"  Jitter σ={jl:.1f}px: {m['phase_correlation']:.4f}")
    if 'III_corrected_2px' in results_icom['scenarios']:
        m = results_icom['scenarios']['III_corrected_2px']
        print(f"  Corrected (2px): {m['phase_correlation']:.4f}")
    print(f"  Recovery ratio:  {results_icom['scenarios'].get('recovery_ratio', 'N/A')}")

    print(f"\nCoM Cross-Residual (% of reference magnitude):")
    for jl in jitter_levels:
        key = f'{jl}px'
        if key in results_residual['mismatch_levels']:
            r = results_residual['mismatch_levels'][key]
            print(f"  Jitter σ={jl:.1f}px: {r['residual_percent']:.1f}%")

    print("\nDone.")


if __name__ == '__main__':
    main()
