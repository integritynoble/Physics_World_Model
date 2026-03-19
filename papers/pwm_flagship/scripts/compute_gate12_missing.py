#!/usr/bin/env python3
"""Compute Gate 1 (information deficiency) and Gate 2 (noise) degradation
for the 5 modalities currently marked 'n.t.' in Figure 3b:
  Fluorescence, Compressive Holography, Cryo-EM, CBCT, Ultrasound.

Methodology matches Tables S12/S13: sweep one parameter to extreme while
keeping the forward model perfectly calibrated (ideal operator).

Output: gate12_missing_results.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.signal import hilbert, fftconvolve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))
RESULTS_DIR = PROJECT_ROOT / "papers" / "pwm_flagship" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.RandomState(42)


def psnr(ref, test, data_range=None):
    ref, test = ref.astype(np.float64), test.astype(np.float64)
    mse = np.mean((ref - test) ** 2)
    if mse < 1e-15:
        return 100.0
    if data_range is None:
        data_range = ref.max() - ref.min()
    if data_range < 1e-15:
        return 0.0
    return float(10 * np.log10(data_range**2 / mse))


def make_phantom(nz=128, nx=128):
    """Generate a simple phantom for testing."""
    phantom = np.zeros((nz, nx), dtype=np.float64)
    cy, cx = nz // 2, nx // 2
    for iz in range(nz):
        for ix in range(nx):
            r = np.sqrt(((iz - cy) / (0.4 * nz)) ** 2 + ((ix - cx) / (0.45 * nx)) ** 2)
            if r < 1.0:
                phantom[iz, ix] = 1.0
            r2 = np.sqrt(((iz - cy - nz // 6) / (0.1 * nz)) ** 2 +
                         ((ix - cx) / (0.15 * nx)) ** 2)
            if r2 < 1.0:
                phantom[iz, ix] = 0.3
    phantom += 0.1 * rng.randn(nz, nx) * (phantom > 0)
    phantom = np.clip(phantom, 0, None)
    return phantom


# =====================================================================
# 1. FLUORESCENCE MICROSCOPY
# =====================================================================
def fluorescence_gate12():
    """G1: PSF sigma sweep (1.5 → 15 px). G2: photon count sweep (1000 → 5)."""
    from pwm_core.physics.microscopy.fluorescence_operator import FluorescenceMicroscopyOperator

    N = 128
    phantom = make_phantom(N, N)
    phantom = phantom / max(phantom.max(), 1e-10)

    true_sigma_ex, true_sigma_em = 1.5, 2.0
    peak_photons_nominal = 1000.0
    n_rl = 80

    def _gauss_psf(sigma, n):
        """Create normalized 2D Gaussian PSF kernel."""
        ax = np.arange(n) - n // 2
        xx, yy = np.meshgrid(ax, ax)
        k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return k / k.sum()

    def run_fluor(sigma_ex, sigma_em, peak_photons):
        """Forward model + Richardson-Lucy reconstruction."""
        op = FluorescenceMicroscopyOperator(
            operator_id="fl_test", nx=N, ny=N,
            psf_sigma_ex=sigma_ex, psf_sigma_em=sigma_em,
            quantum_yield=0.7, background=0.02,
        )
        y_clean = op.forward(phantom)
        # Add Poisson noise
        y_scaled = np.clip(y_clean * peak_photons, 0, None)
        y_noisy = rng.poisson(y_scaled).astype(np.float64) / peak_photons

        # Richardson-Lucy with combined PSF (sigma_total ≈ sqrt(sigma_ex² + sigma_em²))
        sigma_total = np.sqrt(sigma_ex**2 + sigma_em**2)
        psf = _gauss_psf(sigma_total, N)
        psf_flip = psf[::-1, ::-1]
        recon = np.ones_like(phantom) * 0.5
        for _ in range(n_rl):
            predicted = fftconvolve(recon, psf, mode='same') + 1e-12
            ratio = y_noisy / predicted
            correction = fftconvolve(ratio, psf_flip, mode='same')
            recon = recon * correction
            recon = np.clip(recon, 0, None)
        return psnr(phantom, recon)

    psnr_nom = run_fluor(true_sigma_ex, true_sigma_em, peak_photons_nominal)

    # G1: Increase PSF sigma (wider PSF = less resolution = less information)
    g1_sigmas = [3.0, 5.0, 8.0, 12.0, 15.0]
    g1_results = []
    for s in g1_sigmas:
        ratio = true_sigma_em / true_sigma_ex
        p = run_fluor(s, s * ratio, peak_photons_nominal)
        g1_results.append({"sigma": s, "psnr": round(p, 2), "delta": round(p - psnr_nom, 2)})

    # G2: Decrease photon count (more noise)
    g2_photons = [500, 100, 50, 10, 5]
    g2_results = []
    for ph in g2_photons:
        p = run_fluor(true_sigma_ex, true_sigma_em, ph)
        g2_results.append({"photons": ph, "psnr": round(p, 2), "delta": round(p - psnr_nom, 2)})

    return {
        "modality": "Fluorescence",
        "psnr_nominal": round(psnr_nom, 2),
        "g1_parameter": "PSF_sigma (px)",
        "g1_sweep": g1_results,
        "g1_extreme_delta": g1_results[-1]["delta"],
        "g2_parameter": "peak_photons",
        "g2_sweep": g2_results,
        "g2_extreme_delta": g2_results[-1]["delta"],
    }


# =====================================================================
# 2. CRYO-EM
# =====================================================================
def cryoem_gate12():
    """G1: B-factor sweep (2 → 200 Å²). G2: noise sigma sweep (0.05 → 1.0)."""
    from pwm_core.physics.electron.cryoem_operator import CryoEMOperator

    N = 128
    phantom = make_phantom(N, N)
    phantom = phantom / max(phantom.max(), 1e-10)

    defocus_nm = -2000.0
    Cs_mm = 2.0
    pixel_nm = 0.1
    B_nominal = 2.0
    noise_nominal = 0.05
    wiener_snr = 50.0

    def run_cryoem(B_factor, noise_sigma):
        op = CryoEMOperator(
            operator_id="cryo_test", nx=N, ny=N,
            defocus_nm=defocus_nm, Cs_mm=Cs_mm,
            pixel_size_nm=pixel_nm, B_factor=B_factor,
            ice_thickness_nm=50.0,
        )
        micrograph = op.forward(phantom)
        noise = noise_sigma * np.std(micrograph) * rng.randn(*micrograph.shape)
        y_noisy = micrograph + noise

        # Wiener filter using the precomputed transfer function
        Y = np.fft.fft2(y_noisy)
        H = op._transfer  # combined CTF * envelope * ice_atten
        recon = np.real(np.fft.ifft2(np.conj(H) * Y / (np.abs(H)**2 + 1.0/wiener_snr)))
        return psnr(phantom, recon)

    psnr_nom = run_cryoem(B_nominal, noise_nominal)

    # G1: Increase B-factor (damps high-frequency information)
    g1_bfactors = [5.0, 10.0, 20.0, 50.0, 200.0]
    g1_results = []
    for b in g1_bfactors:
        p = run_cryoem(b, noise_nominal)
        g1_results.append({"B_factor": b, "psnr": round(p, 2), "delta": round(p - psnr_nom, 2)})

    # G2: Increase noise
    g2_noise = [0.1, 0.2, 0.5, 0.8, 1.0]
    g2_results = []
    for ns in g2_noise:
        p = run_cryoem(B_nominal, ns)
        g2_results.append({"noise_sigma": ns, "psnr": round(p, 2), "delta": round(p - psnr_nom, 2)})

    return {
        "modality": "Cryo-EM",
        "psnr_nominal": round(psnr_nom, 2),
        "g1_parameter": "B_factor (Å²)",
        "g1_sweep": g1_results,
        "g1_extreme_delta": g1_results[-1]["delta"],
        "g2_parameter": "noise_sigma",
        "g2_sweep": g2_results,
        "g2_extreme_delta": g2_results[-1]["delta"],
    }


# =====================================================================
# 3. CBCT (CT with angle sweep for G1, photon sweep for G2)
# =====================================================================
def cbct_gate12():
    """G1: projection angles sweep (360 → 5). G2: Poisson photon count sweep."""
    from pwm_core.physics.tomography.ct_operator import CTOperator

    N = 128
    phantom = make_phantom(N, N)
    phantom = phantom / max(phantom.max(), 1e-10)

    n_angles_nominal = 360

    def fbp(sinogram, n_angles, N):
        """Simple FBP: Ram-Lak filter + backprojection."""
        n_det = sinogram.shape[1]
        # Ram-Lak filter in frequency domain
        freqs = np.fft.fftfreq(n_det)
        ram_lak = np.abs(freqs)
        # Filter each projection
        filtered = np.zeros_like(sinogram)
        for i in range(n_angles):
            proj_f = np.fft.fft(sinogram[i])
            filtered[i] = np.real(np.fft.ifft(proj_f * ram_lak))
        # Backproject
        angles = np.linspace(0, 180, n_angles, endpoint=False)
        recon = np.zeros((N, N), dtype=np.float64)
        for i, angle in enumerate(angles):
            smeared = np.tile(filtered[i], (N, 1))
            rotated = ndimage.rotate(smeared, -angle, reshape=False, mode='constant', order=1)
            recon += rotated
        return recon * np.pi / (2 * n_angles)

    def run_ct(n_angles, photon_count):
        op = CTOperator(
            operator_id="cbct_test", x_shape=(N, N), n_angles=n_angles,
        )
        sinogram = op.forward(phantom).astype(np.float64)

        if photon_count < 1e10:
            # Poisson noise on projections
            sino_min = sinogram.min()
            sino_shifted = sinogram - sino_min + 0.01
            sino_norm = sino_shifted / sino_shifted.max()
            sino_noisy_counts = rng.poisson(sino_norm * photon_count).astype(np.float64)
            sino_noisy = sino_noisy_counts / photon_count * sino_shifted.max() + sino_min
        else:
            sino_noisy = sinogram

        recon = fbp(sino_noisy, n_angles, N)
        return psnr(phantom, recon)

    psnr_nom = run_ct(n_angles_nominal, int(1e12))

    # G1: fewer projection angles
    g1_angles = [180, 90, 30, 10, 5]
    g1_results = []
    for na in g1_angles:
        p = run_ct(na, int(1e12))
        g1_results.append({"n_angles": na, "psnr": round(p, 2), "delta": round(p - psnr_nom, 2)})

    # G2: lower photon count
    g2_photons = [100000, 10000, 1000, 100, 10]
    g2_results = []
    for ph in g2_photons:
        p = run_ct(n_angles_nominal, ph)
        g2_results.append({"photons": ph, "psnr": round(p, 2), "delta": round(p - psnr_nom, 2)})

    return {
        "modality": "CBCT",
        "psnr_nominal": round(psnr_nom, 2),
        "g1_parameter": "n_angles",
        "g1_sweep": g1_results,
        "g1_extreme_delta": g1_results[-1]["delta"],
        "g2_parameter": "photon_count",
        "g2_sweep": g2_results,
        "g2_extreme_delta": g2_results[-1]["delta"],
    }


# =====================================================================
# 4. COMPRESSIVE HOLOGRAPHY
# =====================================================================
def compholo_gate12():
    """G1: n_depths sweep (4 → 1). G2: noise std sweep (0.01 → 0.5)."""
    from pwm_core.physics.microscopy.compressive_holography_operator import (
        CompressiveHolographyOperator,
    )

    N = 128
    phantom_2d = make_phantom(N, N)
    phantom_2d = phantom_2d / max(phantom_2d.max(), 1e-10)
    # Multi-depth object: phantom at 4 depth planes with slight variations
    n_depths_nominal = 4
    phantom_4d = np.stack([
        phantom_2d * (1.0 - 0.1 * k) + 0.05 * rng.randn(N, N)
        for k in range(n_depths_nominal)
    ])
    phantom_4d = np.clip(phantom_4d, 0, None)

    noise_nominal = 0.01

    def run_holo(n_depths, noise_std):
        ph = phantom_4d[:n_depths]  # use first n_depths planes
        op = CompressiveHolographyOperator(
            operator_id="holo_test", nx=N, ny=N,
            wavelength_nm=532.0, pixel_size_um=5.0,
            carrier_freq=0.15, n_depths=n_depths,
            depth_spacing_um=200.0,
        )
        hologram = op.forward(ph)
        hmax = max(np.abs(hologram).max(), 1e-10)
        noise = noise_std * hmax * rng.randn(*hologram.shape)
        y_noisy = hologram + noise

        # Reconstruct via adjoint (back-propagation)
        recon = op.adjoint(y_noisy)
        recon = np.real(recon) if np.iscomplexobj(recon) else recon

        # Compare first depth plane only
        return psnr(ph[0], recon[0])

    psnr_nom = run_holo(n_depths_nominal, noise_nominal)

    # G1: fewer depth planes (less diversity → less information)
    g1_depths = [3, 2, 1]
    g1_results = []
    for nd in g1_depths:
        p = run_holo(nd, noise_nominal)
        g1_results.append({"n_depths": nd, "psnr": round(p, 2), "delta": round(p - psnr_nom, 2)})

    # G2: more noise
    g2_noise = [0.02, 0.05, 0.1, 0.3, 0.5]
    g2_results = []
    for ns in g2_noise:
        p = run_holo(n_depths_nominal, ns)
        g2_results.append({"noise_std": ns, "psnr": round(p, 2), "delta": round(p - psnr_nom, 2)})

    return {
        "modality": "Comp. Holography",
        "psnr_nominal": round(psnr_nom, 2),
        "g1_parameter": "n_depths",
        "g1_sweep": g1_results,
        "g1_extreme_delta": g1_results[-1]["delta"],
        "g2_parameter": "noise_std",
        "g2_sweep": g2_results,
        "g2_extreme_delta": g2_results[-1]["delta"],
    }


# =====================================================================
# 5. ULTRASOUND
# =====================================================================
def ultrasound_gate12():
    """G1: angle count sweep (75 → 1). G2: additive noise on RF data."""
    from pwm_core.physics.ultrasound.ultrasound_operator import UltrasoundOperator

    N = 128
    n_elements = 128
    n_samples = 2048
    fs = 100e6
    true_sos = 1540.0
    n_angles_nominal = 75
    pitch = 0.3e-3

    phantom = make_phantom(N, N)
    phantom = phantom / max(phantom.max(), 1e-10)

    # Generate synthetic multi-angle RF data
    op = UltrasoundOperator(
        operator_id="us_test", nz=N, nx=N,
        n_elements=n_elements, n_samples=n_samples,
        speed_of_sound=true_sos, fs=fs,
    )
    rf_single = op.forward(phantom)
    all_angles = np.linspace(np.radians(-16), np.radians(16), n_angles_nominal)

    rf_75 = np.zeros((n_angles_nominal, n_elements, n_samples), dtype=np.float64)
    for ai, angle in enumerate(all_angles):
        shift_samples = int(np.sin(angle) * n_elements * pitch / true_sos * fs * 0.5)
        rf_75[ai] = np.roll(rf_single, shift_samples, axis=1)

    def compound_das(rf_data, angles, c, nz=128, nx=128):
        n_ang = rf_data.shape[0]
        n_elem = rf_data.shape[1]
        n_samp = rf_data.shape[2]
        aperture = n_elem * pitch
        x_pos = np.linspace(-aperture / 2, aperture / 2, nx)
        depth = (n_samp / fs) * c / 2
        z_pos = np.linspace(0.5 * depth / nz, depth - 0.5 * depth / nz, nz)
        elem_x = np.linspace(-aperture / 2, aperture / 2, n_elem)
        Z, X = np.meshgrid(z_pos, x_pos, indexing="ij")
        rx_dist = np.sqrt(Z[None, :, :] ** 2 + (X[None, :, :] - elem_x[:, None, None]) ** 2)
        image = np.zeros((nz, nx), dtype=np.float64)
        for ai in range(n_ang):
            t_tx = (Z * np.cos(angles[ai]) + X * np.sin(angles[ai])) / c
            t_total = t_tx[None, :, :] + rx_dist / c
            idx = (t_total * fs).astype(np.int64)
            valid = (idx >= 0) & (idx < n_samp)
            idx_safe = np.clip(idx, 0, n_samp - 1)
            ie = np.arange(n_elem)[:, None, None]
            gathered = rf_data[ai][ie, idx_safe] * valid
            image += gathered.sum(axis=0)
        analytic = hilbert(image, axis=0)
        return np.abs(analytic)

    # Nominal reconstruction
    recon_nom = compound_das(rf_75, all_angles, true_sos, N, N)
    recon_nom -= recon_nom.min()
    if recon_nom.max() > 1e-12:
        recon_nom /= recon_nom.max()
    pseudo_gt = recon_nom.copy()

    def self_ref_psnr(recon):
        r = recon.copy()
        r -= r.min()
        if r.max() > 1e-12:
            r /= r.max()
        s = np.dot(pseudo_gt.ravel(), r.ravel()) / max(np.dot(r.ravel(), r.ravel()), 1e-15)
        r_scaled = r * s
        return psnr(pseudo_gt, r_scaled)

    # G1: Fewer angles
    g1_angles_list = [37, 15, 7, 3, 1]
    g1_results = []
    for na in g1_angles_list:
        indices = np.round(np.linspace(0, n_angles_nominal - 1, na)).astype(int)
        rf_sub = rf_75[indices]
        angles_sub = all_angles[indices]
        recon = compound_das(rf_sub, angles_sub, true_sos, N, N)
        p = self_ref_psnr(recon)
        g1_results.append({"n_angles": na, "psnr": round(p, 2), "delta": round(p - 100.0, 2)})

    # G2: Add noise to RF data
    rf_std = np.std(rf_75)
    g2_noise_fracs = [0.01, 0.05, 0.1, 0.3, 0.5]
    g2_results = []
    for nf in g2_noise_fracs:
        rf_noisy = rf_75 + nf * rf_std * rng.randn(*rf_75.shape)
        recon = compound_das(rf_noisy, all_angles, true_sos, N, N)
        p = self_ref_psnr(recon)
        g2_results.append({"noise_frac": nf, "psnr": round(p, 2), "delta": round(p - 100.0, 2)})

    return {
        "modality": "Ultrasound",
        "psnr_nominal": 100.0,
        "g1_parameter": "n_angles",
        "g1_sweep": g1_results,
        "g1_extreme_delta": g1_results[-1]["delta"],
        "g2_parameter": "RF_noise_fraction",
        "g2_sweep": g2_results,
        "g2_extreme_delta": g2_results[-1]["delta"],
    }


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    results = {}

    modalities = [
        ("Fluorescence", fluorescence_gate12),
        ("Cryo-EM", cryoem_gate12),
        ("CBCT", cbct_gate12),
        ("Comp. Holography", compholo_gate12),
        ("Ultrasound", ultrasound_gate12),
    ]

    for name, func in modalities:
        print(f"\n{'='*60}")
        print(f"Computing G1/G2 for {name}...")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            r = func()
            dt = time.time() - t0
            r["compute_time_s"] = round(dt, 1)
            results[name] = r
            print(f"  Nominal PSNR: {r['psnr_nominal']:.2f} dB")
            print(f"  G1 extreme: {r['g1_extreme_delta']:+.1f} dB")
            print(f"  G2 extreme: {r['g2_extreme_delta']:+.1f} dB")
            print(f"  Time: {dt:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    out_path = RESULTS_DIR / "gate12_missing_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: G1/G2 extreme degradation (dB)")
    print(f"{'='*60}")
    print(f"{'Modality':<20} {'G1 (extreme)':<15} {'G2 (extreme)':<15}")
    print("-" * 50)
    for name in ["Fluorescence", "Cryo-EM", "CBCT", "Comp. Holography", "Ultrasound"]:
        r = results.get(name, {})
        g1 = r.get("g1_extreme_delta", "ERR")
        g2 = r.get("g2_extreme_delta", "ERR")
        g1s = f"{g1:+.1f}" if isinstance(g1, (int, float)) else str(g1)
        g2s = f"{g2:+.1f}" if isinstance(g2, (int, float)) else str(g2)
        print(f"{name:<20} {g1s:<15} {g2s:<15}")
