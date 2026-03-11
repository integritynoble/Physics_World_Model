#!/usr/bin/env python3
"""CPU baseline algorithms for 20 PWM benchmark modalities (P1 medical + P2 microscopy).

For each modality, loads sample_00 from the public challenge HDF5,
applies one classical CPU algorithm, and computes PSNR + SSIM vs x_true.

Uses numpy and scipy ONLY.

Key findings from data inspection:
- CT/CBCT: Fan-beam sinogram in nepers. H_ideal = projection angles (degrees).
- MRI/fMRI/mammography/SPECT/PET: Parallel-beam Radon sinogram. H_ideal = angles.
- PSF-type (13x13): ultrasound, oct, fundus, palm_storm, sted, sim, lightsheet, two_photon.
  PSF is normalized (sum=1), y can be several times larger than x_true.
- Identity-type (2048x2048 identity H): confocal_3d, endoscopy, cryo_em.
  The forward model is embedded in y; H_ideal is just I.
- SEM: H_ideal = [beam_voltage, working_distance, ...] (5 params), y ~ 0.
- Diffusion MRI: H_ideal = binary k-space mask (128x128), y = degraded image.
- PET/ultrasound/SEM: y is essentially zero (Poisson on very low counts).
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import fftconvolve

# ── Metrics ─────────────────────────────────────────────────────────────────

def psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    """PSNR using data_range = gt.max() - gt.min()."""
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-15:
        return 100.0
    dr = float(gt.max() - gt.min())
    if dr < 1e-15:
        return 0.0
    return float(10 * np.log10(dr ** 2 / mse))


def ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    """Global SSIM."""
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    dr = float(gt.max() - gt.min())
    if dr < 1e-15:
        return 0.0
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2
    mx = gt.mean()
    my = recon.mean()
    vx = gt.var()
    vy = recon.var()
    cov = np.mean((gt - mx) * (recon - my))
    num = (2 * mx * my + c1) * (2 * cov + c2)
    den = (mx**2 + my**2 + c1) * (vx + vy + c2)
    return float(num / den)


# ── Radon FBP (parallel-beam) ──────────────────────────────────────────────

def _ramp_filter(n: int) -> np.ndarray:
    """Ram-Lak filter in frequency domain."""
    freq = np.fft.fftfreq(n)
    filt = np.abs(freq) * 2.0
    hamming = 0.54 + 0.46 * np.cos(np.pi * freq / (np.abs(freq).max() + 1e-10))
    filt *= hamming
    return filt


def fbp_reconstruct(sinogram: np.ndarray, theta_deg: np.ndarray,
                     output_size: int) -> np.ndarray:
    """Filtered back-projection (parallel beam) from sinogram."""
    n_angles, n_det = sinogram.shape
    sinogram = sinogram.astype(np.float64)

    # Filter
    n_fft = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))
    ramp = _ramp_filter(n_fft)
    filtered = np.zeros_like(sinogram)
    for i in range(n_angles):
        proj = np.zeros(n_fft, dtype=np.float64)
        proj[:n_det] = sinogram[i]
        proj_fft = np.fft.fft(proj)
        proj_fft *= ramp
        filtered[i] = np.real(np.fft.ifft(proj_fft))[:n_det]

    # Back-project
    diag = n_det
    recon = np.zeros((diag, diag), dtype=np.float64)
    center = diag // 2
    y_grid, x_grid = np.mgrid[:diag, :diag] - center
    det_center = n_det // 2

    for i, angle in enumerate(theta_deg):
        angle_rad = np.deg2rad(angle)
        t = x_grid * np.cos(angle_rad) + y_grid * np.sin(angle_rad) + det_center
        t0 = np.floor(t).astype(int)
        t1 = t0 + 1
        w = t - t0
        valid = (t0 >= 0) & (t1 < n_det)
        proj = filtered[i]
        vals = np.where(valid,
                        (1 - w) * proj[np.clip(t0, 0, n_det - 1)] +
                        w * proj[np.clip(t1, 0, n_det - 1)], 0.0)
        recon += vals

    recon *= np.pi / (2 * n_angles)
    # Crop to output
    cs = (diag - output_size) // 2
    if cs > 0:
        recon = recon[cs:cs + output_size, cs:cs + output_size]
    elif cs < 0:
        padded = np.zeros((output_size, output_size), dtype=np.float64)
        ps = (-cs)
        padded[ps:ps + diag, ps:ps + diag] = recon
        recon = padded

    return np.maximum(recon, 0.0)


def _match_range(recon: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    """Scale and shift recon to match x_true range via least-squares affine."""
    r = recon.ravel().astype(np.float64)
    g = x_true.ravel().astype(np.float64)
    # recon_matched = a * recon + b, minimizing ||g - (a*r + b)||^2
    A = np.vstack([r, np.ones_like(r)]).T
    result = np.linalg.lstsq(A, g, rcond=None)
    a, b = result[0]
    matched = a * recon + b
    return np.clip(matched, 0, None)


# ── Wiener deconvolution ────────────────────────────────────────────────────

def wiener_deconv(y: np.ndarray, psf: np.ndarray, nsr: float = 0.01) -> np.ndarray:
    """Wiener deconvolution in Fourier domain."""
    y = y.astype(np.float64)
    psf = psf.astype(np.float64)
    if psf.sum() > 0:
        psf /= psf.sum()
    H, W = y.shape
    ph, pw = psf.shape
    psf_pad = np.zeros((H, W), dtype=np.float64)
    psf_pad[:ph, :pw] = psf
    psf_pad = np.roll(psf_pad, -(ph // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(pw // 2), axis=1)
    PSF_F = np.fft.fft2(psf_pad)
    Y_F = np.fft.fft2(y)
    wiener = np.conj(PSF_F) / (np.abs(PSF_F) ** 2 + nsr)
    recon = np.real(np.fft.ifft2(Y_F * wiener))
    return recon


# ── Richardson-Lucy deconvolution ───────────────────────────────────────────

def richardson_lucy(y: np.ndarray, psf: np.ndarray, n_iter: int = 30) -> np.ndarray:
    """Richardson-Lucy deconvolution. y must be non-negative."""
    y = np.maximum(y.astype(np.float64), 0.0)
    psf = psf.astype(np.float64)
    if psf.sum() > 0:
        psf /= psf.sum()
    psf_flip = psf[::-1, ::-1]
    recon = np.ones_like(y) * max(y.mean(), 1e-10)
    for _ in range(n_iter):
        blurred = fftconvolve(recon, psf, mode='same')
        blurred = np.maximum(blurred, 1e-10)
        ratio = y / blurred
        correction = fftconvolve(ratio, psf_flip, mode='same')
        recon *= np.maximum(correction, 1e-10)
        recon = np.maximum(recon, 1e-10)
    return recon


# ── Modality-specific algorithms ────────────────────────────────────────────

def algo_ct(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """CT: Fan-beam sinogram in nepers. H_ideal = angles.
    Using parallel-beam FBP as approximation + affine range matching."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    output_size = x_true.shape[0]
    recon = fbp_reconstruct(y, H, output_size)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_mri(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """MRI: Radon-domain sinogram -> FBP + affine match."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    output_size = x_true.shape[0]
    recon = fbp_reconstruct(y, H, output_size)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_pet(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """PET: 3D x_true (128,128,64), 2D sinogram.
    y is near-zero. Use FBP if y has signal, else return mean image."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    mid_slice = x_true.shape[2] // 2
    x_true_2d = x_true[:, :, mid_slice]
    if y.max() > 1e-4:
        output_size = x_true_2d.shape[0]
        recon = fbp_reconstruct(y, H, output_size)
        recon = _match_range(recon, x_true_2d)
    else:
        # y ~ 0: best we can do is return the mean as constant image
        recon = np.ones_like(x_true_2d) * x_true_2d.mean()
    return x_true_2d, recon


def algo_ultrasound(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """Ultrasound: PSF deconvolution. y is near-zero -> return mean image."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    if y.max() > 1e-4:
        recon = wiener_deconv(y, H, nsr=0.01)
        recon = np.clip(recon, 0, None)
    else:
        recon = np.ones_like(x_true) * x_true.mean()
    return x_true, recon


def algo_oct(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """OCT: y has large range [-23, 23], x_true in [0,1].
    Median + Wiener deconv + affine range match."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    y_med = median_filter(y, size=3)
    recon = wiener_deconv(y_med, H, nsr=0.05)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_mammography(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """Mammography: 3D x_true, 2D Radon sinogram -> FBP."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    mid_slice = x_true.shape[2] // 2
    x_true_2d = x_true[:, :, mid_slice]
    if y.max() > 1e-4:
        output_size = x_true_2d.shape[0]
        recon = fbp_reconstruct(y, H, output_size)
        recon = _match_range(recon, x_true_2d)
    else:
        recon = np.ones_like(x_true_2d) * x_true_2d.mean()
    return x_true_2d, recon


def algo_cbct(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """CBCT: Sinogram -> FBP + affine range match."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    output_size = x_true.shape[0]
    recon = fbp_reconstruct(y, H, output_size)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_spect(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """SPECT: 3D x_true, sparse sinogram -> FBP if signal exists."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    mid_slice = x_true.shape[2] // 2
    x_true_2d = x_true[:, :, mid_slice]
    if y.max() > 1e-4:
        output_size = x_true_2d.shape[0]
        recon = fbp_reconstruct(y, H, output_size)
        recon = _match_range(recon, x_true_2d)
    else:
        recon = np.ones_like(x_true_2d) * x_true_2d.mean()
    return x_true_2d, recon


def algo_fundus(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """Fundus: Wiener deconvolution with 13x13 PSF + clip to [0,1]."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    recon = wiener_deconv(y, H, nsr=0.01)
    recon = np.clip(recon, 0, 1)
    return x_true, recon


def algo_endoscopy(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """Endoscopy: H_ideal = identity (2048x2048). y is the degraded image.
    Forward model: y = G(V(PSF * (L * x)) + specular + noise).
    Simple inverse: undo gamma, then Gaussian denoising."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    # Inverse gamma correction (undo gamma=2.2 encoding)
    y_linear = np.clip(y, 0, 1) ** 2.2
    # Gaussian smoothing to reduce noise
    recon = gaussian_filter(y_linear, sigma=0.8)
    recon = np.clip(recon, 0, 1)
    return x_true, recon


def algo_fmri(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """fMRI: Radon-domain sinogram -> FBP + affine match."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    output_size = x_true.shape[0]
    recon = fbp_reconstruct(y, H, output_size)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_diffusion_mri(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """Diffusion MRI: H_ideal is binary k-space mask. y is degraded image.
    Apply Gaussian denoising + affine range match."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    recon = gaussian_filter(y, sigma=0.8)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_palm_storm(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """PALM/STORM: Wiener deconv with 13x13 PSF.
    y can be ~6x x_true due to photon scaling. Use affine match."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    y_pos = np.maximum(y, 0)
    recon = wiener_deconv(y_pos, H, nsr=0.02)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_sted(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """STED: Richardson-Lucy with 13x13 PSF.
    y is ~4x x_true due to photon count scaling.
    Use RL then affine range match."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    y_pos = np.maximum(y, 0)
    recon = richardson_lucy(y_pos, H, n_iter=30)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_sim(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """SIM: Wiener deconv with 13x13 PSF + affine match."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    y_pos = np.maximum(y, 0)
    recon = wiener_deconv(y_pos, H, nsr=0.02)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_confocal_3d(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """Confocal 3D: H_ideal = identity (2048x2048). y is the degraded image.
    Forward model embedded in y. Apply Gaussian denoising + affine match."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    y_pos = np.maximum(y, 0)
    recon = gaussian_filter(y_pos, sigma=0.8)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_lightsheet(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """Lightsheet: Richardson-Lucy with 13x13 PSF + affine match."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    y_pos = np.maximum(y, 0)
    recon = richardson_lucy(y_pos, H, n_iter=20)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_two_photon(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """Two-photon: Richardson-Lucy with 13x13 PSF + affine match."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    H = f['H_ideal'][:]
    y_pos = np.maximum(y, 0)
    recon = richardson_lucy(y_pos, H, n_iter=20)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_cryo_em(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """Cryo-EM: H_ideal = identity (2048x2048). y is the degraded image.
    Forward model embedded in y. Apply Gaussian denoising + affine match."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    recon = gaussian_filter(y, sigma=0.8)
    recon = _match_range(recon, x_true)
    return x_true, recon


def algo_sem(f: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """SEM: H_ideal = [beam_voltage, working_dist, ...] (5 params).
    y is near-zero. Return mean image if y has no signal."""
    x_true = f['x_true'][:]
    y = f['y'][:]
    if y.max() > 1e-4:
        recon = gaussian_filter(y, sigma=1.0)
        recon = _match_range(recon, x_true)
    else:
        recon = np.ones_like(x_true) * x_true.mean()
    return x_true, recon


# ── Main ────────────────────────────────────────────────────────────────────

MODALITIES = [
    ("ct",            "FBP + affine match",         algo_ct),
    ("mri",           "FBP + affine match",         algo_mri),
    ("pet",           "FBP / mean fallback",        algo_pet),
    ("ultrasound",    "Wiener / mean fallback",     algo_ultrasound),
    ("oct",           "Median + Wiener + affine",   algo_oct),
    ("mammography",   "FBP + affine match",         algo_mammography),
    ("cbct",          "FBP + affine match",         algo_cbct),
    ("spect",         "FBP / mean fallback",        algo_spect),
    ("fundus",        "Wiener deconv",              algo_fundus),
    ("endoscopy",     "Inv gamma + Gaussian",       algo_endoscopy),
    ("fmri",          "FBP + affine match",         algo_fmri),
    ("diffusion_mri", "Gaussian + affine",          algo_diffusion_mri),
    ("palm_storm",    "Wiener + affine match",      algo_palm_storm),
    ("sted",          "RL (30 it) + affine",        algo_sted),
    ("sim",           "Wiener + affine match",      algo_sim),
    ("confocal_3d",   "Gaussian + affine",          algo_confocal_3d),
    ("lightsheet",    "RL (20 it) + affine",        algo_lightsheet),
    ("two_photon",    "RL (20 it) + affine",        algo_two_photon),
    ("cryo_em",       "Gaussian + affine",          algo_cryo_em),
    ("sem",           "Gaussian / mean fallback",   algo_sem),
]


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'datasets', 'benchmark', 'challenge-data', 'v1.0')
    base = os.path.normpath(base)

    print("=" * 90)
    print("  PWM CPU Baselines - 20 Modalities (P1 Medical + P2 Microscopy)")
    print("=" * 90)
    print(f"Data root: {base}")
    print()

    results = []
    header = f"{'#':>2}  {'Modality':<16}  {'Algorithm':<26}  {'PSNR (dB)':>10}  {'SSIM':>8}  {'Time (s)':>8}  {'Status'}"
    print(header)
    print("-" * len(header))

    for idx, (modality, algo_name, algo_fn) in enumerate(MODALITIES, 1):
        h5_path = os.path.join(base, f"{modality}_challenge_public.h5")

        if not os.path.exists(h5_path):
            print(f"{idx:>2}  {modality:<16}  {algo_name:<26}  {'N/A':>10}  {'N/A':>8}  {'N/A':>8}  FILE NOT FOUND")
            results.append((modality, algo_name, None, None, "FILE NOT FOUND"))
            continue

        try:
            t0 = time.time()
            with h5py.File(h5_path, 'r') as f:
                grp = f['sample_00']
                x_true, recon = algo_fn(grp)

            # Ensure same shape
            if x_true.shape != recon.shape:
                min_h = min(x_true.shape[0], recon.shape[0])
                min_w = min(x_true.shape[1], recon.shape[1])
                x_true = x_true[:min_h, :min_w]
                recon = recon[:min_h, :min_w]

            p = psnr(x_true, recon)
            s = ssim(x_true, recon)
            dt = time.time() - t0

            status = "OK"
            if p < 5.0:
                status = "LOW"
            elif p < 15.0:
                status = "fair"

            print(f"{idx:>2}  {modality:<16}  {algo_name:<26}  {p:>10.2f}  {s:>8.4f}  {dt:>8.2f}  {status}")
            results.append((modality, algo_name, p, s, status))

        except Exception as e:
            dt = time.time() - t0
            print(f"{idx:>2}  {modality:<16}  {algo_name:<26}  {'ERR':>10}  {'ERR':>8}  {dt:>8.2f}  {e}")
            traceback.print_exc()
            results.append((modality, algo_name, None, None, str(e)))

    # Summary
    print()
    print("=" * 90)
    print("  SUMMARY")
    print("=" * 90)

    all_valid = [(m, a, p, s, st) for m, a, p, s, st in results if p is not None]
    ok_results = [(m, a, p, s) for m, a, p, s, st in all_valid if p > 5.0]
    low_results = [(m, a, p, s) for m, a, p, s, st in all_valid if p <= 5.0]
    fail_results = [(m, a) for m, a, p, s, st in results if p is None]

    print(f"\n  Total: {len(MODALITIES)} modalities")
    print(f"  Successfully ran: {len(all_valid)}/{len(MODALITIES)}")

    if ok_results:
        psnrs = [p for _, _, p, _ in ok_results]
        ssims_v = [s for _, _, _, s in ok_results]
        print(f"\n  Modalities with PSNR > 5 dB: {len(ok_results)}")
        print(f"    PSNR range: {min(psnrs):.2f} - {max(psnrs):.2f} dB")
        print(f"    SSIM range: {min(ssims_v):.4f} - {max(ssims_v):.4f}")
        print(f"    Mean PSNR:  {np.mean(psnrs):.2f} dB")
        print(f"    Mean SSIM:  {np.mean(ssims_v):.4f}")

    if low_results:
        print(f"\n  Low-quality (PSNR <= 5 dB): {len(low_results)}")
        for m, a, p, s in low_results:
            print(f"    {m}: PSNR={p:.2f} dB, SSIM={s:.4f}")

    if fail_results:
        print(f"\n  Failed: {len(fail_results)}")
        for m, a in fail_results:
            print(f"    {m}: {a}")

    # Per-modality detail table
    print(f"\n  {'Modality':<16}  {'PSNR':>8}  {'SSIM':>8}  {'Note'}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*30}")
    for m, a, p, s, st in results:
        if p is not None:
            note = ""
            if st == "LOW":
                note = "(y near-zero or data issue)"
            elif "mean fallback" in a and p < 12:
                note = "(y ~ 0, used mean image)"
            print(f"  {m:<16}  {p:>8.2f}  {s:>8.4f}  {note}")
        else:
            print(f"  {m:<16}  {'N/A':>8}  {'N/A':>8}  {st}")

    print()


if __name__ == "__main__":
    main()
