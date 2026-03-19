#!/usr/bin/env python3
"""
Generate benchmark datasets for 11 optical/fluorescence microscopy modalities:
  tirf, flim, expansion, ism, minflux, confocal_livecell,
  confocal_endomicroscopy, spinning_disk, three_photon, lattice_lightsheet, clem

Each modality: 3 tiers (public=12, dev=20, hidden=20 samples)
Output: datasets/benchmark/{modality}/{tier}/{modality}_challenge_{tier}.h5
        + spec.json, true_spec.json, images/sample_NN/
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import fftconvolve

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

ROOT = Path(__file__).resolve().parent.parent
BENCH_BASE = ROOT / "datasets" / "benchmark"
IMAGE_SIZE = 256

# ── Utility: PSF / phantom / metrics ─────────────────────────────────────────

def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        size += 1
    h = size // 2
    yy, xx = np.mgrid[-h:h+1, -h:h+1].astype(np.float64)
    psf = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    psf /= psf.sum()
    return psf.astype(np.float32)


def make_cell_phantom(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Shepp-Logan style cell phantom."""
    x = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[:H, :W]
    n = rng.integers(5, 15)
    for _ in range(n):
        cy, cx = rng.uniform(H*0.1, H*0.9), rng.uniform(W*0.1, W*0.9)
        a = rng.uniform(8, 30); b = a * rng.uniform(0.5, 1.0)
        angle = rng.uniform(0, np.pi)
        intensity = rng.uniform(0.3, 1.0)
        dy, dx = yy - cy, xx - cx
        ca, sa = np.cos(angle), np.sin(angle)
        r2 = ((dy*ca + dx*sa)/a)**2 + ((-dy*sa + dx*ca)/b)**2
        mask = r2 < 1.0
        texture = gaussian_filter(rng.uniform(0, 1, (H, W)), sigma=rng.uniform(2, 5))
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        x[mask] += intensity * (0.6 + 0.4 * texture[mask])
    return x


def make_blob_phantom(H: int, W: int, rng: np.random.Generator, n_blobs: int = 20) -> np.ndarray:
    """Random Gaussian blobs."""
    x = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[:H, :W].astype(np.float64)
    for _ in range(n_blobs):
        cy, cx = rng.uniform(10, H-10), rng.uniform(10, W-10)
        sigma = rng.uniform(3, 15)
        amp = rng.uniform(0.3, 1.0)
        x += amp * np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*sigma**2))
    return x


def make_filament_phantom(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Thin filament network."""
    x = np.zeros((H, W), dtype=np.float64)
    for _ in range(rng.integers(15, 35)):
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        angle = rng.uniform(0, 2*np.pi)
        speed = rng.uniform(1.0, 3.0)
        thickness = rng.uniform(0.5, 1.5)
        intensity = rng.uniform(0.4, 1.0)
        pts_y, pts_x = [cy], [cx]
        for _ in range(rng.integers(30, 80)):
            angle += rng.normal(0, 0.15)
            pts_y.append(pts_y[-1] + speed*np.sin(angle))
            pts_x.append(pts_x[-1] + speed*np.cos(angle))
        for py, px in zip(pts_y, pts_x):
            iy, ix = int(round(py)), int(round(px))
            r = max(1, int(np.ceil(thickness*2.5)))
            y0, y1 = max(0, iy-r), min(H, iy+r+1)
            x0, x1 = max(0, ix-r), min(W, ix+r+1)
            if y1 > y0 and x1 > x0:
                yyg = np.arange(y0, y1)[:, None].astype(np.float64)
                xxg = np.arange(x0, x1)[None, :].astype(np.float64)
                d2 = (yyg-py)**2 + (xxg-px)**2
                x[y0:y1, x0:x1] += intensity * np.exp(-d2/(2*thickness**2))
    return x


def make_point_sources(H: int, W: int, rng: np.random.Generator, n: int = 50) -> np.ndarray:
    """Sparse point sources for SMLM-like phantoms."""
    x = np.zeros((H, W), dtype=np.float64)
    for _ in range(n):
        iy = rng.integers(5, H-5)
        ix = rng.integers(5, W-5)
        amp = rng.uniform(0.5, 1.0)
        sigma = rng.uniform(0.5, 1.5)
        r = 3
        y0, y1 = max(0, iy-r), min(H, iy+r+1)
        x0, x1 = max(0, ix-r), min(W, ix+r+1)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        x[y0:y1, x0:x1] += amp * np.exp(-((yy-iy)**2 + (xx-ix)**2)/(2*sigma**2))
    return x


def normalize_01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def add_poisson(signal: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Scale signal to photon counts, apply Poisson noise, return normalized."""
    lam = np.maximum(signal.astype(np.float64) * scale, 0.01)
    noisy = rng.poisson(lam).astype(np.float64)
    return noisy


def rl_deconv(y: np.ndarray, psf: np.ndarray, n_iter: int = 10) -> np.ndarray:
    """Richardson-Lucy deconvolution."""
    y64 = y.astype(np.float64)
    psf64 = psf.astype(np.float64)
    psf_flip = psf64[::-1, ::-1]
    x_est = np.full_like(y64, 0.5)
    x_est = np.maximum(x_est, 1e-8)
    for _ in range(n_iter):
        denom = fftconvolve(x_est, psf64, mode='same') + 1e-8
        ratio = y64 / denom
        correction = fftconvolve(ratio, psf_flip, mode='same')
        x_est *= correction
        x_est = np.maximum(x_est, 1e-8)
    return normalize_01(x_est)


def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((gt.astype(np.float64) - recon.astype(np.float64))**2))
    if mse < 1e-12:
        return 100.0
    dr = float(gt.max() - gt.min())
    if dr < 1e-12:
        return 0.0
    return float(10.0 * np.log10(dr**2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    g, r = gt.astype(np.float64), recon.astype(np.float64)
    dr = float(g.max() - g.min())
    if dr < 1e-12:
        return 0.0
    c1, c2 = (0.01*dr)**2, (0.03*dr)**2
    mu_g, mu_r = g.mean(), r.mean()
    cov = float(np.mean((g-mu_g)*(r-mu_r)))
    return float(((2*mu_g*mu_r+c1)*(2*cov+c2)) /
                 ((mu_g**2+mu_r**2+c1)*(g.var()+r.var()+c2)))


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def save_png(arr: np.ndarray, path: Path) -> None:
    if not HAS_PIL:
        return
    if arr.ndim == 3:
        # Multi-channel: take first channel for PNG
        arr = arr[0]
    img = np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img, "L").save(str(path))


def save_png_color(arr_rgb: np.ndarray, path: Path) -> None:
    """Save (3, H, W) or (H, W, 3) as color PNG."""
    if not HAS_PIL:
        return
    if arr_rgb.ndim == 3 and arr_rgb.shape[0] == 3:
        arr_rgb = arr_rgb.transpose(1, 2, 0)
    arr_rgb = np.clip(_norm(arr_rgb) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr_rgb, "RGB").save(str(path))


def write_tier(
    modality: str,
    tier: str,
    n_samples: int,
    seed_offset: int,
    mismatch_seed: int,
    spec_ranges: dict,
    forward_fn,       # fn(x_true, mis, rng) -> (y, H_ideal, extra_arrays, recon)
    phantom_fn,       # fn(idx, seed) -> x_true
    forward_desc: str,
    baseline_desc: str,
) -> None:
    tier_dir = BENCH_BASE / modality / tier
    images_dir = tier_dir / "images"
    tier_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"{modality}_challenge_{tier}.h5"
    rng_mis = np.random.default_rng(mismatch_seed)
    true_specs: dict = {}
    psnrs, ssims = [], []

    print(f"\n[{modality}/{tier}] -> {n_samples} samples -> {h5_path}")

    with h5py.File(h5_path, "w") as f:
        f.attrs["modality"] = modality
        f.attrs["tier"] = tier
        f.attrs["n_samples"] = n_samples
        f.attrs["image_size"] = IMAGE_SIZE
        f.attrs["forward_model"] = forward_desc
        f.attrs["baseline_method"] = baseline_desc
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)

        for idx in range(n_samples):
            key = f"sample_{idx:02d}"
            x_true = phantom_fn(idx, seed_offset + idx)

            # Sample mismatch params
            mis = {}
            for k, v in spec_ranges.items():
                val = float(rng_mis.uniform(v["min"], v["max"]))
                mis[k] = val
            true_specs[key] = {**mis, "phantom_seed": seed_offset + idx}

            sample_rng = np.random.default_rng(mismatch_seed + idx + 1)
            result = forward_fn(x_true, mis, sample_rng)
            y, H_ideal, extra, recon = result

            # Compute metrics vs x_true for reconstruction quality
            # x_true might be multi-channel; use first channel or mean
            gt_for_metric = x_true if x_true.ndim == 2 else x_true[0]
            recon_for_metric = recon if recon.ndim == 2 else recon[0]
            if gt_for_metric.shape != recon_for_metric.shape:
                # Different shapes (e.g. ISM), skip metrics
                psnr, ssim = 0.0, 0.0
            else:
                psnr = compute_psnr(gt_for_metric, recon_for_metric)
                ssim = compute_ssim(gt_for_metric, recon_for_metric)
            psnrs.append(psnr)
            ssims.append(ssim)

            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true, compression="gzip")
            grp.create_dataset("y", data=y, compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")
            for extra_key, extra_val in extra.items():
                grp.create_dataset(extra_key, data=extra_val, compression="gzip")

            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["metadata"] = json.dumps({
                "shapes": {
                    "x_true": list(x_true.shape),
                    "y": list(y.shape),
                    "H_ideal": list(H_ideal.shape),
                    "reconstruction_baseline": list(recon.shape),
                },
                "psnr_baseline_db": round(psnr, 2),
                "ssim_baseline": round(ssim, 4),
            })

            # Images
            sample_img_dir = images_dir / f"sample_{idx:02d}"
            sample_img_dir.mkdir(parents=True, exist_ok=True)
            # Choose the 2D slice for x_true PNG
            if x_true.ndim == 2:
                save_png(x_true, sample_img_dir / "x_true.png")
            elif x_true.ndim == 3 and x_true.shape[0] >= 2:
                # CLEM-style: save as two separate images
                save_png(x_true[0], sample_img_dir / "x_true_ch0.png")
                save_png(x_true[1], sample_img_dir / "x_true_ch1.png")
            if y.ndim == 2:
                save_png(y, sample_img_dir / "y_measurement.png")
            elif y.ndim == 3:
                save_png(y[0], sample_img_dir / "y_measurement_ch0.png")
            if H_ideal.ndim == 2:
                save_png(H_ideal, sample_img_dir / "H_ideal.png")
            if recon.ndim == 2:
                save_png(recon, sample_img_dir / "reconstruction.png")

            with open(sample_img_dir / "spec.json", "w") as sf:
                json.dump({"true_spec": mis, "spec_ranges": spec_ranges,
                           "psnr_db": psnr, "ssim": ssim}, sf, indent=2)

            if idx == 0 or (idx + 1) % 5 == 0:
                print(f"  {key}: x_true{list(x_true.shape)} y{list(y.shape)} "
                      f"H_ideal{list(H_ideal.shape)} recon{list(recon.shape)} "
                      f"PSNR={psnr:.1f}dB SSIM={ssim:.3f}")

        f.attrs["mean_psnr_baseline_db"] = float(np.mean(psnrs)) if psnrs else 0.0
        f.attrs["mean_ssim_baseline"] = float(np.mean(ssims)) if ssims else 0.0

    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    sz_mb = os.path.getsize(h5_path) / 1e6
    print(f"  [{modality}/{tier}] Done | mean PSNR={np.mean(psnrs):.1f}dB "
          f"SSIM={np.mean(ssims):.3f} | {sz_mb:.1f} MB")


# ═════════════════════════════════════════════════════════════════════════════
# 1. TIRF
# ═════════════════════════════════════════════════════════════════════════════

TIRF_SPEC = {
    "public":  {"penetration_depth_nm": {"min": 50, "max": 300, "unit": "nm"},
                "wavelength_nm": {"min": 488, "max": 647, "unit": "nm"},
                "NA": {"min": 1.40, "max": 1.49, "unit": ""}},
    "dev":     {"penetration_depth_nm": {"min": 50, "max": 300, "unit": "nm"},
                "wavelength_nm": {"min": 488, "max": 647, "unit": "nm"},
                "NA": {"min": 1.40, "max": 1.49, "unit": ""}},
    "hidden":  {"penetration_depth_nm": {"min": 50, "max": 300, "unit": "nm"},
                "wavelength_nm": {"min": 488, "max": 647, "unit": "nm"},
                "NA": {"min": 1.40, "max": 1.49, "unit": ""}},
}

PSF_TIRF_SIZE = 11
PSF_TIRF_SIGMA = 1.5


def tirf_phantom(idx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fns = [make_cell_phantom, make_blob_phantom, make_filament_phantom]
    x = fns[idx % 3](IMAGE_SIZE, IMAGE_SIZE, rng)
    # Near-membrane: emphasize edge/surface layer with evanescent envelope
    yy = np.linspace(0, 1, IMAGE_SIZE)[:, None]
    surface_weight = np.exp(-yy * 3.0)  # membrane at top
    x = x * (0.4 + 0.6 * surface_weight)
    return normalize_01(x)


def tirf_forward(x_true: np.ndarray, mis: dict, rng: np.random.Generator):
    # Evanescent field decay (penetration depth modulates intensity)
    d_nm = mis["penetration_depth_nm"]
    # Normalize penetration (50nm=strong confinement, 300nm=weaker)
    decay = d_nm / 300.0  # 0.17..1.0
    yy = np.linspace(0, 1, IMAGE_SIZE)[:, None]
    evanescent = np.exp(-yy / (decay + 0.05))

    signal = x_true * evanescent

    # PSF convolution
    psf = gaussian_psf(PSF_TIRF_SIZE, PSF_TIRF_SIGMA)
    convolved = fftconvolve(signal, psf, mode='same').astype(np.float32)
    convolved = np.clip(convolved, 0, None)

    H_ideal = normalize_01(convolved)

    # Poisson noise
    scale = 1000.0
    noisy = add_poisson(convolved / (convolved.max() + 1e-8), scale, rng)
    noisy /= (noisy.max() + 1e-8)
    y = noisy.astype(np.float32)

    # RL deconv baseline
    recon = rl_deconv(y, psf, n_iter=10)

    return y, H_ideal, {}, recon


def generate_tirf():
    for tier, n_s, s_off, m_seed in [("public", 12, 0, 1000),
                                      ("dev", 20, 10000, 11000),
                                      ("hidden", 20, 20000, 21000)]:
        write_tier(
            "tirf", tier, n_s, s_off, m_seed,
            TIRF_SPEC[tier],
            tirf_forward,
            tirf_phantom,
            "y = Poisson(evanescent_field(d_nm) * PSF_gaussian(sigma=1.5) * x_true); "
            "H_ideal: noiseless PSF-convolved evanescent-weighted image",
            "Richardson-Lucy deconvolution 10 iterations",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 2. FLIM
# ═════════════════════════════════════════════════════════════════════════════

FLIM_SPEC = {
    "public":  {"gate_width_ns": {"min": 0.5, "max": 2.0, "unit": "ns"},
                "repetition_rate_MHz": {"min": 10, "max": 80, "unit": "MHz"},
                "photons_per_pixel": {"min": 10, "max": 1000, "unit": "photons"}},
    "dev":     {"gate_width_ns": {"min": 0.5, "max": 2.0, "unit": "ns"},
                "repetition_rate_MHz": {"min": 10, "max": 80, "unit": "MHz"},
                "photons_per_pixel": {"min": 10, "max": 1000, "unit": "photons"}},
    "hidden":  {"gate_width_ns": {"min": 0.5, "max": 2.0, "unit": "ns"},
                "repetition_rate_MHz": {"min": 10, "max": 80, "unit": "MHz"},
                "photons_per_pixel": {"min": 10, "max": 1000, "unit": "photons"}},
}


def flim_phantom(idx: int, seed: int) -> np.ndarray:
    """Lifetime map: range [0,1] representing 0.5-5 ns."""
    rng = np.random.default_rng(seed)
    # Different regions with different lifetimes
    x = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
    # Background lifetime (~0.2 normalized = 1ns)
    x[:] = 0.2
    fns = [make_cell_phantom, make_blob_phantom, make_filament_phantom]
    structure = fns[idx % 3](IMAGE_SIZE, IMAGE_SIZE, rng)
    # Map structure intensity to lifetime variation
    x += structure * 0.7
    return normalize_01(x)


def flim_forward(x_true: np.ndarray, mis: dict, rng: np.random.Generator):
    """
    Time-gated intensity: I(gate) = integral of I0*exp(-t/tau) over gate window.
    Two gates: early [0, gate_width] and late [gate_width, 2*gate_width].
    y = ratio of early/late gates (proxy for lifetime).
    """
    gate_w = mis["gate_width_ns"]
    photons = mis["photons_per_pixel"]

    # Lifetime map: tau in [0.5, 5] ns
    tau_map = 0.5 + x_true.astype(np.float64) * 4.5  # [0.5, 5] ns

    # Gate 1: [0, gate_w]; Gate 2: [gate_w, 2*gate_w]
    # Integral of exp(-t/tau) dt from t1 to t2 = tau * (exp(-t1/tau) - exp(-t2/tau))
    gate1 = tau_map * (1.0 - np.exp(-gate_w / tau_map))
    gate2 = tau_map * (np.exp(-gate_w / tau_map) - np.exp(-2*gate_w / tau_map))

    # Scale by photons
    gate1_counts = np.maximum(gate1 / (gate1.max() + 1e-8) * photons, 0.01)
    gate2_counts = np.maximum(gate2 / (gate2.max() + 1e-8) * photons, 0.01)

    # Poisson noise on each gate
    g1_noisy = rng.poisson(gate1_counts).astype(np.float64) + 0.01
    g2_noisy = rng.poisson(gate2_counts).astype(np.float64) + 0.01

    # Combined time-gated image (weighted sum)
    combined = g1_noisy + 0.5 * g2_noisy
    y = normalize_01(combined).astype(np.float32)

    # H_ideal: ones (FLIM uses uniform spatial sensitivity)
    H_ideal = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    # Baseline: ratio of two time-gate windows (lifetime estimator)
    ratio = g1_noisy / (g1_noisy + g2_noisy + 1e-8)
    recon = normalize_01(ratio).astype(np.float32)

    return y, H_ideal, {}, recon


def generate_flim():
    for tier, n_s, s_off, m_seed in [("public", 12, 0, 2000),
                                      ("dev", 20, 10000, 12000),
                                      ("hidden", 20, 20000, 22000)]:
        write_tier(
            "flim", tier, n_s, s_off, m_seed,
            FLIM_SPEC[tier],
            flim_forward,
            flim_phantom,
            "y = time-gated intensity (Poisson); two-gate exponential decay model; "
            "H_ideal = ones (uniform sensitivity)",
            "Ratio of two time-gate windows (rapid lifetime determination)",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. Expansion Microscopy
# ═════════════════════════════════════════════════════════════════════════════

EXPANSION_SPEC = {
    "public":  {"expansion_factor": {"min": 3, "max": 10, "unit": "x"},
                "gel_concentration": {"min": 1, "max": 5, "unit": "%"},
                "antibody_rounds": {"min": 1, "max": 3, "unit": "rounds"}},
    "dev":     {"expansion_factor": {"min": 3, "max": 10, "unit": "x"},
                "gel_concentration": {"min": 1, "max": 5, "unit": "%"},
                "antibody_rounds": {"min": 1, "max": 3, "unit": "rounds"}},
    "hidden":  {"expansion_factor": {"min": 3, "max": 10, "unit": "x"},
                "gel_concentration": {"min": 1, "max": 5, "unit": "%"},
                "antibody_rounds": {"min": 1, "max": 3, "unit": "rounds"}},
}

PSF_EXPANSION_SIZE = 5
PSF_EXPANSION_SIGMA = 0.7  # tight PSF for post-expansion


def expansion_phantom(idx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fns = [make_cell_phantom, make_blob_phantom, make_filament_phantom]
    x = fns[idx % 3](IMAGE_SIZE, IMAGE_SIZE, rng)
    return normalize_01(x)


def expansion_forward(x_true: np.ndarray, mis: dict, rng: np.random.Generator):
    """
    Simulate post-expansion image: tight PSF (equivalent to expanded resolution)
    with Poisson noise. y simulates what you'd see post-physical expansion.
    """
    factor = mis["expansion_factor"]
    # Simulate that expansion improves effective resolution by factor
    # Effective PSF sigma in original pixel units = 1/factor (tighter)
    eff_sigma = max(0.5, PSF_EXPANSION_SIGMA * (4.0 / factor))
    psf = gaussian_psf(PSF_EXPANSION_SIZE, eff_sigma)

    convolved = fftconvolve(x_true.astype(np.float64), psf, mode='same')
    convolved = np.clip(convolved, 0, None)
    H_ideal = gaussian_psf(PSF_EXPANSION_SIZE, PSF_EXPANSION_SIGMA).astype(np.float32)

    # Poisson noise
    noisy = add_poisson(convolved / (convolved.max() + 1e-8), 800.0, rng)
    noisy /= (noisy.max() + 1e-8)
    y = noisy.astype(np.float32)

    # RL deconv baseline
    psf_nominal = gaussian_psf(PSF_EXPANSION_SIZE, PSF_EXPANSION_SIGMA)
    recon = rl_deconv(y, psf_nominal, n_iter=20)

    return y, H_ideal, {}, recon


def generate_expansion():
    for tier, n_s, s_off, m_seed in [("public", 12, 0, 3000),
                                      ("dev", 20, 10000, 13000),
                                      ("hidden", 20, 20000, 23000)]:
        write_tier(
            "expansion", tier, n_s, s_off, m_seed,
            EXPANSION_SPEC[tier],
            expansion_forward,
            expansion_phantom,
            "y = Poisson(PSF_tight(sigma=f(expansion_factor)) * x_true); "
            "H_ideal: (5,5) tight PSF kernel",
            "Richardson-Lucy deconvolution 20 iterations",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4. ISM (Image Scanning Microscopy)
# ═════════════════════════════════════════════════════════════════════════════

ISM_SPEC = {
    "public":  {"pinhole_airy_units": {"min": 0.5, "max": 2.0, "unit": "AU"},
                "scan_step_nm": {"min": 10, "max": 100, "unit": "nm"},
                "NA": {"min": 0.8, "max": 1.4, "unit": ""}},
    "dev":     {"pinhole_airy_units": {"min": 0.5, "max": 2.0, "unit": "AU"},
                "scan_step_nm": {"min": 10, "max": 100, "unit": "nm"},
                "NA": {"min": 0.8, "max": 1.4, "unit": ""}},
    "hidden":  {"pinhole_airy_units": {"min": 0.5, "max": 2.0, "unit": "AU"},
                "scan_step_nm": {"min": 10, "max": 100, "unit": "nm"},
                "NA": {"min": 0.8, "max": 1.4, "unit": ""}},
}

ISM_DET = 5   # 5x5 detector array
ISM_SUB = 64  # sub-image size per detector element


def ism_phantom(idx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fns = [make_cell_phantom, make_blob_phantom, make_filament_phantom]
    x = fns[idx % 3](IMAGE_SIZE, IMAGE_SIZE, rng)
    return normalize_01(x)


def ism_forward(x_true: np.ndarray, mis: dict, rng: np.random.Generator):
    """
    5x5 detector array; each element captures shifted sub-image.
    y: (25, 64, 64) float32
    H_ideal: (25, 2) float32 - detector offset positions (in pixels)
    reconstruction_baseline: pixel reassignment -> (256,256)
    """
    n_det = ISM_DET * ISM_DET  # 25
    sub = ISM_SUB

    # Downsample x_true to sub-image size
    x_down = zoom(x_true, sub / IMAGE_SIZE, order=1)  # (64, 64)

    pinhole = mis["pinhole_airy_units"]
    psf_sigma = 1.5 * pinhole  # effective PSF sigma scales with pinhole

    psf = gaussian_psf(11, psf_sigma)

    det_offsets = []
    sub_images = np.zeros((n_det, sub, sub), dtype=np.float32)

    det_idx = 0
    half = ISM_DET // 2
    for di in range(-half, half+1):
        for dj in range(-half, half+1):
            off_y = di * (sub / IMAGE_SIZE) * 2  # offset in sub-image pixels
            off_x = dj * (sub / IMAGE_SIZE) * 2
            det_offsets.append([float(off_y), float(off_x)])

            # Shift phantom slightly per detector element
            from scipy.ndimage import shift as ndimage_shift
            x_shifted = ndimage_shift(x_down, [off_y, off_x], mode='reflect')
            blurred = fftconvolve(x_shifted, psf, mode='same')
            blurred = np.clip(blurred, 0, None)
            noisy = rng.poisson(np.maximum(blurred / (blurred.max() + 1e-8) * 500, 0.01))
            sub_images[det_idx] = normalize_01(noisy.astype(np.float64)).astype(np.float32)
            det_idx += 1

    y = sub_images  # (25, 64, 64)
    H_ideal = np.array(det_offsets, dtype=np.float32)  # (25, 2)

    # Baseline: pixel reassignment (sum sub-images after back-shifting, upsample to 256x256)
    recon_64 = np.zeros((sub, sub), dtype=np.float64)
    from scipy.ndimage import shift as ndimage_shift
    for k, (oy, ox) in enumerate(det_offsets):
        shifted_back = ndimage_shift(sub_images[k].astype(np.float64), [-oy*0.5, -ox*0.5])
        recon_64 += shifted_back

    # Upsample to 256x256
    recon_256 = zoom(recon_64, IMAGE_SIZE / sub, order=3)
    recon = normalize_01(recon_256).astype(np.float32)

    return y, H_ideal, {}, recon


def generate_ism():
    for tier, n_s, s_off, m_seed in [("public", 12, 0, 4000),
                                      ("dev", 20, 10000, 14000),
                                      ("hidden", 20, 20000, 24000)]:
        write_tier(
            "ism", tier, n_s, s_off, m_seed,
            ISM_SPEC[tier],
            ism_forward,
            ism_phantom,
            "y = (25, 64, 64) 5x5 detector array sub-images; each shifted by detector offset; "
            "H_ideal = (25, 2) detector offset positions in pixels",
            "Pixel reassignment: sum shifted sub-images, upsample to 256x256",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 5. MINFLUX
# ═════════════════════════════════════════════════════════════════════════════

MINFLUX_SPEC = {
    "public":  {"lambda_donut_nm": {"min": 20, "max": 50, "unit": "nm"},
                "photons_per_localization": {"min": 100, "max": 10000, "unit": "photons"},
                "bg_photons": {"min": 0, "max": 100, "unit": "photons"}},
    "dev":     {"lambda_donut_nm": {"min": 20, "max": 50, "unit": "nm"},
                "photons_per_localization": {"min": 100, "max": 10000, "unit": "photons"},
                "bg_photons": {"min": 0, "max": 100, "unit": "photons"}},
    "hidden":  {"lambda_donut_nm": {"min": 20, "max": 50, "unit": "nm"},
                "photons_per_localization": {"min": 100, "max": 10000, "unit": "photons"},
                "bg_photons": {"min": 0, "max": 100, "unit": "photons"}},
}


def minflux_phantom(idx: int, seed: int) -> np.ndarray:
    """SMLM-like: single molecules rendered as Gaussians."""
    rng = np.random.default_rng(seed)
    n_mols = rng.integers(30, 120)
    x = make_point_sources(IMAGE_SIZE, IMAGE_SIZE, rng, n=n_mols)
    return normalize_01(x)


def minflux_forward(x_true: np.ndarray, mis: dict, rng: np.random.Generator):
    bg = mis["bg_photons"]
    photons = mis["photons_per_localization"]
    lambda_nm = mis["lambda_donut_nm"]

    # PSF: Gaussian with sigma related to donut size
    sigma_px = max(0.5, lambda_nm / 50.0 * 2.0)  # scale: 0.8..2.0 px
    psf = gaussian_psf(11, sigma_px)

    # Convolve with PSF + background
    signal = fftconvolve(x_true.astype(np.float64), psf, mode='same')
    signal = np.clip(signal, 0, None)
    signal = signal / (signal.max() + 1e-8) * photons + bg

    noisy = rng.poisson(np.maximum(signal, 0.01)).astype(np.float64)
    noisy /= (noisy.max() + 1e-8)
    y = noisy.astype(np.float32)

    H_ideal = psf.astype(np.float32)

    # Baseline: threshold + Gaussian smoothing
    threshold = float(np.percentile(y, 90))
    recon = np.where(y > threshold, y, 0.0)
    recon = gaussian_filter(recon.astype(np.float64), sigma=0.5)
    recon = normalize_01(recon).astype(np.float32)

    return y, H_ideal, {}, recon


def generate_minflux():
    for tier, n_s, s_off, m_seed in [("public", 12, 0, 5000),
                                      ("dev", 20, 10000, 15000),
                                      ("hidden", 20, 20000, 25000)]:
        write_tier(
            "minflux", tier, n_s, s_off, m_seed,
            MINFLUX_SPEC[tier],
            minflux_forward,
            minflux_phantom,
            "y = Poisson(PSF_donut * x_true * photons + bg); "
            "H_ideal: (11,11) donut PSF",
            "Thresholding + Gaussian smoothing",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 6. Confocal Live Cell
# ═════════════════════════════════════════════════════════════════════════════

CONFOCAL_LIVECELL_SPEC = {
    "public":  {"scan_speed_fps": {"min": 0.5, "max": 30, "unit": "fps"},
                "NA": {"min": 1.2, "max": 1.4, "unit": ""},
                "pixel_dwell_us": {"min": 2, "max": 100, "unit": "us"}},
    "dev":     {"scan_speed_fps": {"min": 0.5, "max": 30, "unit": "fps"},
                "NA": {"min": 1.2, "max": 1.4, "unit": ""},
                "pixel_dwell_us": {"min": 2, "max": 100, "unit": "us"}},
    "hidden":  {"scan_speed_fps": {"min": 0.5, "max": 30, "unit": "fps"},
                "NA": {"min": 1.2, "max": 1.4, "unit": ""},
                "pixel_dwell_us": {"min": 2, "max": 100, "unit": "us"}},
}

PSF_CLC_SIZE = 11
PSF_CLC_SIGMA = 2.0


def confocal_livecell_phantom(idx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fns = [make_cell_phantom, make_blob_phantom, make_filament_phantom]
    x = fns[idx % 3](IMAGE_SIZE, IMAGE_SIZE, rng)
    return normalize_01(x)


def confocal_livecell_forward(x_true: np.ndarray, mis: dict, rng: np.random.Generator):
    scan_speed = mis["scan_speed_fps"]
    na = mis["NA"]
    dwell = mis["pixel_dwell_us"]

    # PSF sigma depends on NA (higher NA = tighter PSF)
    psf_sigma = max(0.8, PSF_CLC_SIGMA * (1.35 / na))
    psf = gaussian_psf(PSF_CLC_SIZE, psf_sigma)

    # Photobleaching gradient (scanline-dependent)
    # Faster scan = less bleaching
    bleach_rate = max(0.0, 1.0 - scan_speed / 30.0) * 0.4  # 0..0.4
    yy = np.linspace(1.0, 1.0 - bleach_rate, IMAGE_SIZE)[:, None]
    bleach_factor = yy * np.ones((1, IMAGE_SIZE))

    signal = x_true.astype(np.float64) * bleach_factor

    # Confocal PSF convolution
    convolved = fftconvolve(signal, psf, mode='same')
    convolved = np.clip(convolved, 0, None)

    # Photon count scales with dwell time
    scale = max(100, dwell * 20)
    noisy = add_poisson(convolved / (convolved.max() + 1e-8), scale, rng)
    noisy /= (noisy.max() + 1e-8)
    y = noisy.astype(np.float32)

    H_ideal = psf.astype(np.float32)

    # RL deconv baseline
    recon = rl_deconv(y, psf, n_iter=10)

    return y, H_ideal, {}, recon


def generate_confocal_livecell():
    for tier, n_s, s_off, m_seed in [("public", 12, 0, 6000),
                                      ("dev", 20, 10000, 16000),
                                      ("hidden", 20, 20000, 26000)]:
        write_tier(
            "confocal_livecell", tier, n_s, s_off, m_seed,
            CONFOCAL_LIVECELL_SPEC[tier],
            confocal_livecell_forward,
            confocal_livecell_phantom,
            "y = Poisson(PSF_confocal(NA) * x_true * bleach_factor(scan_speed)); "
            "H_ideal: (11,11) Gaussian PSF",
            "Richardson-Lucy deconvolution 10 iterations",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 7. Confocal Endomicroscopy
# ═════════════════════════════════════════════════════════════════════════════

CONFOCAL_ENDO_SPEC = {
    "public":  {"fiber_diameter_um": {"min": 2, "max": 5, "unit": "um"},
                "working_distance_mm": {"min": 0, "max": 5, "unit": "mm"},
                "NA_fiber": {"min": 0.35, "max": 0.65, "unit": ""}},
    "dev":     {"fiber_diameter_um": {"min": 2, "max": 5, "unit": "um"},
                "working_distance_mm": {"min": 0, "max": 5, "unit": "mm"},
                "NA_fiber": {"min": 0.35, "max": 0.65, "unit": ""}},
    "hidden":  {"fiber_diameter_um": {"min": 2, "max": 5, "unit": "um"},
                "working_distance_mm": {"min": 0, "max": 5, "unit": "mm"},
                "NA_fiber": {"min": 0.35, "max": 0.65, "unit": ""}},
}


def make_fiber_bundle_mask(H: int, W: int, fiber_spacing: int = 8) -> np.ndarray:
    """Hexagonal fiber bundle mask."""
    mask = np.zeros((H, W), dtype=np.float64)
    fiber_radius = max(1, fiber_spacing // 3)
    yy, xx = np.mgrid[:H, :W].astype(np.float64)

    # Hexagonal grid
    row = 0
    for cy in range(0, H + fiber_spacing, fiber_spacing):
        col_offset = (fiber_spacing // 2) if (row % 2 == 1) else 0
        for cx in range(col_offset, W + fiber_spacing, fiber_spacing):
            d2 = (yy - cy)**2 + (xx - cx)**2
            mask[d2 < fiber_radius**2] = 1.0
        row += 1
    return mask.astype(np.float32)


def confocal_endo_phantom(idx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fns = [make_cell_phantom, make_blob_phantom, make_filament_phantom]
    x = fns[idx % 3](IMAGE_SIZE, IMAGE_SIZE, rng)
    return normalize_01(x)


def confocal_endo_forward(x_true: np.ndarray, mis: dict, rng: np.random.Generator):
    fiber_d = mis["fiber_diameter_um"]
    na_f = mis["NA_fiber"]

    # Fiber spacing proportional to diameter (pixels: ~8-16)
    fiber_spacing = max(6, int(fiber_d * 3))
    fiber_mask = make_fiber_bundle_mask(IMAGE_SIZE, IMAGE_SIZE, fiber_spacing)

    # PSF sigma depends on fiber NA
    psf_sigma = max(1.0, 2.5 / na_f)
    psf = gaussian_psf(11, psf_sigma)

    # Apply fiber mask + PSF
    masked = x_true.astype(np.float64) * fiber_mask
    convolved = fftconvolve(masked, psf, mode='same')
    convolved = np.clip(convolved, 0, None)

    # H_ideal is the fiber bundle mask
    H_ideal = fiber_mask.astype(np.float32)

    noisy = add_poisson(convolved / (convolved.max() + 1e-8), 600.0, rng)
    noisy /= (noisy.max() + 1e-8)
    y = noisy.astype(np.float32)

    # Baseline: interpolate gaps + RL deconv
    # Simple interpolation: Gaussian blur to fill in gaps
    filled = gaussian_filter(y.astype(np.float64), sigma=fiber_spacing // 3)
    recon = rl_deconv(normalize_01(filled).astype(np.float32), psf, n_iter=5)

    return y, H_ideal, {}, recon


def generate_confocal_endomicroscopy():
    for tier, n_s, s_off, m_seed in [("public", 12, 0, 7000),
                                      ("dev", 20, 10000, 17000),
                                      ("hidden", 20, 20000, 27000)]:
        write_tier(
            "confocal_endomicroscopy", tier, n_s, s_off, m_seed,
            CONFOCAL_ENDO_SPEC[tier],
            confocal_endo_forward,
            confocal_endo_phantom,
            "y = Poisson((x_true * fiber_mask) * PSF_gaussian(NA_fiber)); "
            "H_ideal: (256,256) fiber bundle pattern mask",
            "Gaussian interpolation to fill fiber gaps + RL deconvolution 5 iter",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 8. Spinning Disk Confocal
# ═════════════════════════════════════════════════════════════════════════════

SPINNING_DISK_SPEC = {
    "public":  {"disk_rpm": {"min": 1000, "max": 8000, "unit": "rpm"},
                "pinhole_um": {"min": 25, "max": 70, "unit": "um"},
                "NA": {"min": 0.75, "max": 1.45, "unit": ""}},
    "dev":     {"disk_rpm": {"min": 1000, "max": 8000, "unit": "rpm"},
                "pinhole_um": {"min": 25, "max": 70, "unit": "um"},
                "NA": {"min": 0.75, "max": 1.45, "unit": ""}},
    "hidden":  {"disk_rpm": {"min": 1000, "max": 8000, "unit": "rpm"},
                "pinhole_um": {"min": 25, "max": 70, "unit": "um"},
                "NA": {"min": 0.75, "max": 1.45, "unit": ""}},
}

PSF_SD_SIZE = 11
PSF_SD_SIGMA = 1.5


def spinning_disk_phantom(idx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fns = [make_cell_phantom, make_blob_phantom, make_filament_phantom]
    x = fns[idx % 3](IMAGE_SIZE, IMAGE_SIZE, rng)
    return normalize_01(x)


def spinning_disk_forward(x_true: np.ndarray, mis: dict, rng: np.random.Generator):
    na = mis["NA"]
    pinhole = mis["pinhole_um"]

    # PSF sigma: tighter PSF with higher NA, wider with larger pinhole
    psf_sigma = max(0.8, PSF_SD_SIGMA * (1.2 / na) * (pinhole / 50.0))
    psf = gaussian_psf(PSF_SD_SIZE, psf_sigma)

    convolved = fftconvolve(x_true.astype(np.float64), psf, mode='same')
    convolved = np.clip(convolved, 0, None)

    H_ideal = gaussian_psf(PSF_SD_SIZE, PSF_SD_SIGMA).astype(np.float32)

    # Spinning disk: lower noise than single-beam (lower scale gives less shot noise variation)
    scale = 800.0
    noisy = add_poisson(convolved / (convolved.max() + 1e-8), scale, rng)
    noisy /= (noisy.max() + 1e-8)
    y = noisy.astype(np.float32)

    recon = rl_deconv(y, psf, n_iter=10)

    return y, H_ideal, {}, recon


def generate_spinning_disk():
    for tier, n_s, s_off, m_seed in [("public", 12, 0, 8000),
                                      ("dev", 20, 10000, 18000),
                                      ("hidden", 20, 20000, 28000)]:
        write_tier(
            "spinning_disk", tier, n_s, s_off, m_seed,
            SPINNING_DISK_SPEC[tier],
            spinning_disk_forward,
            spinning_disk_phantom,
            "y = Poisson(PSF_gaussian(NA, pinhole) * x_true); "
            "H_ideal: (11,11) Gaussian PSF (sigma=1.5)",
            "Richardson-Lucy deconvolution 10 iterations",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 9. Three-Photon Microscopy
# ═════════════════════════════════════════════════════════════════════════════

THREE_PHOTON_SPEC = {
    "public":  {"wavelength_nm": {"min": 1300, "max": 1700, "unit": "nm"},
                "pulse_energy_nJ": {"min": 0.1, "max": 10, "unit": "nJ"},
                "depth_scattering_mfp": {"min": 1, "max": 5, "unit": "mfp"}},
    "dev":     {"wavelength_nm": {"min": 1300, "max": 1700, "unit": "nm"},
                "pulse_energy_nJ": {"min": 0.1, "max": 10, "unit": "nJ"},
                "depth_scattering_mfp": {"min": 1, "max": 5, "unit": "mfp"}},
    "hidden":  {"wavelength_nm": {"min": 1300, "max": 1700, "unit": "nm"},
                "pulse_energy_nJ": {"min": 0.1, "max": 10, "unit": "nJ"},
                "depth_scattering_mfp": {"min": 1, "max": 5, "unit": "mfp"}},
}

PSF_3P_SIZE = 7
PSF_3P_SIGMA = 0.8  # tight 3P PSF


def three_photon_phantom(idx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fns = [make_cell_phantom, make_blob_phantom, make_filament_phantom]
    x = fns[idx % 3](IMAGE_SIZE, IMAGE_SIZE, rng)
    return normalize_01(x)


def three_photon_forward(x_true: np.ndarray, mis: dict, rng: np.random.Generator):
    depth_mfp = mis["depth_scattering_mfp"]
    pulse_nJ = mis["pulse_energy_nJ"]

    # Depth attenuation (exponential; simulate deep-tissue effect)
    # Rows = depth; attenuation increases with depth
    depth_atten = np.exp(-np.linspace(0, depth_mfp, IMAGE_SIZE) / 3.0)[:, None]
    depth_factor = depth_atten * np.ones((1, IMAGE_SIZE))

    # 3P signal scales as I^3 (cubic dependence on intensity)
    x_attenuated = x_true.astype(np.float64) * depth_factor

    psf = gaussian_psf(PSF_3P_SIZE, PSF_3P_SIGMA)

    convolved = fftconvolve(x_attenuated, psf, mode='same')
    convolved = np.clip(convolved, 0, None)

    H_ideal = psf.astype(np.float32)

    # Shot noise; scale by pulse energy (more energy = more signal)
    scale = max(200, pulse_nJ * 200)
    noisy = add_poisson(convolved / (convolved.max() + 1e-8), scale, rng)
    noisy /= (noisy.max() + 1e-8)
    y = noisy.astype(np.float32)

    recon = rl_deconv(y, psf, n_iter=10)

    return y, H_ideal, {}, recon


def generate_three_photon():
    for tier, n_s, s_off, m_seed in [("public", 12, 0, 9000),
                                      ("dev", 20, 10000, 19000),
                                      ("hidden", 20, 20000, 29000)]:
        write_tier(
            "three_photon", tier, n_s, s_off, m_seed,
            THREE_PHOTON_SPEC[tier],
            three_photon_forward,
            three_photon_phantom,
            "y = Poisson(PSF_tight(sigma=0.8) * x_true * depth_attenuation(mfp)); "
            "H_ideal: (7,7) tight PSF",
            "Richardson-Lucy deconvolution 10 iterations",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 10. Lattice Light Sheet
# ═════════════════════════════════════════════════════════════════════════════

LATTICE_LS_SPEC = {
    "public":  {"NA_excitation": {"min": 0.4, "max": 0.6, "unit": ""},
                "zeta_lattice_period_um": {"min": 0.4, "max": 0.5, "unit": "um"},
                "dithering": {"min": 0, "max": 1, "unit": "bool"}},
    "dev":     {"NA_excitation": {"min": 0.4, "max": 0.6, "unit": ""},
                "zeta_lattice_period_um": {"min": 0.4, "max": 0.5, "unit": "um"},
                "dithering": {"min": 0, "max": 1, "unit": "bool"}},
    "hidden":  {"NA_excitation": {"min": 0.4, "max": 0.6, "unit": ""},
                "zeta_lattice_period_um": {"min": 0.4, "max": 0.5, "unit": "um"},
                "dithering": {"min": 0, "max": 1, "unit": "bool"}},
}


def make_anisotropic_psf(H: int, W: int, sigma_xy: float, sigma_z: float) -> np.ndarray:
    """Anisotropic PSF: elongated along vertical axis (z-direction)."""
    if H % 2 == 0: H += 1
    if W % 2 == 0: W += 1
    hy, hx = H // 2, W // 2
    yy, xx = np.mgrid[-hy:hy+1, -hx:hx+1].astype(np.float64)
    psf = np.exp(-(xx**2 / (2*sigma_xy**2) + yy**2 / (2*sigma_z**2)))
    psf /= psf.sum()
    return psf.astype(np.float32)


def lattice_ls_phantom(idx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fns = [make_cell_phantom, make_blob_phantom, make_filament_phantom]
    x = fns[idx % 3](IMAGE_SIZE, IMAGE_SIZE, rng)
    return normalize_01(x)


def lattice_ls_forward(x_true: np.ndarray, mis: dict, rng: np.random.Generator):
    na_exc = mis["NA_excitation"]
    dithering = mis["dithering"] > 0.5

    # Anisotropic PSF: tight laterally, elongated axially (z)
    sigma_xy = max(0.8, 1.5 * (0.5 / na_exc))
    sigma_z = sigma_xy * 3.0  # light sheet elongated in z

    psf = make_anisotropic_psf(11, 31, sigma_xy, sigma_z)  # tall PSF
    convolved = fftconvolve(x_true.astype(np.float64), psf, mode='same')
    convolved = np.clip(convolved, 0, None)

    # Stripe artifact along x direction (lattice pattern)
    if not dithering:
        period = max(4, int(mis["zeta_lattice_period_um"] * 20))  # ~8-10 px
        stripe = 0.05 * np.sin(2 * np.pi * np.arange(IMAGE_SIZE) / period)[None, :]
        convolved += stripe * convolved.max()

    H_ideal = make_anisotropic_psf(IMAGE_SIZE, IMAGE_SIZE, sigma_xy, sigma_z).astype(np.float32)

    noisy = add_poisson(convolved / (convolved.max() + 1e-8), 800.0, rng)
    noisy /= (noisy.max() + 1e-8)
    y = noisy.astype(np.float32)

    # Baseline: Fourier notch destripe (notch at stripe frequency) + RL
    # Simple: high-pass along x to remove stripes, then RL
    # Use FFT-based approach: suppress low-freq in kx but not ky
    Y = np.fft.rfft2(y.astype(np.float64))
    # Notch filter: suppress periodic stripe at specific kx frequency
    kx = np.fft.rfftfreq(IMAGE_SIZE)
    stripe_freq = 1.0 / period if not dithering else 0.1
    for i, k in enumerate(kx):
        if abs(abs(k) - stripe_freq) < stripe_freq * 0.3:
            Y[:, i] *= 0.1
    y_destriped = np.fft.irfft2(Y, s=(IMAGE_SIZE, IMAGE_SIZE))
    y_destriped = normalize_01(y_destriped).astype(np.float32)

    # RL with small PSF kernel
    psf_small = gaussian_psf(11, sigma_xy)
    recon = rl_deconv(y_destriped, psf_small, n_iter=8)

    return y, H_ideal, {}, recon


def generate_lattice_lightsheet():
    for tier, n_s, s_off, m_seed in [("public", 12, 0, 30000),
                                      ("dev", 20, 10000, 40000),
                                      ("hidden", 20, 20000, 50000)]:
        write_tier(
            "lattice_lightsheet", tier, n_s, s_off, m_seed,
            LATTICE_LS_SPEC[tier],
            lattice_ls_forward,
            lattice_ls_phantom,
            "y = Poisson(anisotropic_PSF(NA_exc) * x_true) + stripe_artifact(no dithering); "
            "H_ideal: (256,256) anisotropic PSF elongated along z",
            "Fourier notch destripe + Richardson-Lucy deconvolution",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 11. CLEM (Correlative Light-Electron Microscopy)
# ═════════════════════════════════════════════════════════════════════════════

CLEM_SPEC = {
    "public":  {"registration_error_px": {"min": 0.5, "max": 5, "unit": "px"},
                "pixel_ratio": {"min": 5, "max": 20, "unit": "ratio"},
                "staining_contrast": {"min": 0.5, "max": 2.0, "unit": ""}},
    "dev":     {"registration_error_px": {"min": 0.5, "max": 5, "unit": "px"},
                "pixel_ratio": {"min": 5, "max": 20, "unit": "ratio"},
                "staining_contrast": {"min": 0.5, "max": 2.0, "unit": ""}},
    "hidden":  {"registration_error_px": {"min": 0.5, "max": 5, "unit": "px"},
                "pixel_ratio": {"min": 5, "max": 20, "unit": "ratio"},
                "staining_contrast": {"min": 0.5, "max": 2.0, "unit": ""}},
}


def clem_phantom(idx: int, seed: int) -> np.ndarray:
    """(2, 256, 256) float32: [FM_channel, EM_channel]."""
    rng = np.random.default_rng(seed)
    # FM channel: fluorescence labeling (cell-like)
    fm = make_cell_phantom(IMAGE_SIZE, IMAGE_SIZE, rng)
    fm = normalize_01(fm)

    # EM channel: ultrastructure (fine detail, high contrast)
    rng2 = np.random.default_rng(seed + 500000)
    em_fine = make_filament_phantom(IMAGE_SIZE, IMAGE_SIZE, rng2)
    em_coarse = make_cell_phantom(IMAGE_SIZE, IMAGE_SIZE, rng2)
    em = 0.5 * normalize_01(em_fine) + 0.5 * normalize_01(em_coarse)
    em = normalize_01(em)

    return np.stack([fm, em], axis=0).astype(np.float32)  # (2, 256, 256)


def clem_forward(x_true: np.ndarray, mis: dict, rng: np.random.Generator):
    """
    x_true: (2, 256, 256) [FM, EM]
    y: (256, 256) registered overlay
    H_ideal: (3, 3) affine registration matrix
    recon: y itself
    """
    from scipy.ndimage import shift as ndimage_shift

    reg_err = mis["registration_error_px"]
    pixel_ratio = mis["pixel_ratio"]
    staining = mis["staining_contrast"]

    fm = x_true[0].astype(np.float64)
    em = x_true[1].astype(np.float64)

    # Apply registration error to FM channel
    shift_y = rng.uniform(-reg_err, reg_err)
    shift_x = rng.uniform(-reg_err, reg_err)
    fm_shifted = ndimage_shift(fm, [shift_y, shift_x], mode='reflect')

    # EM channel: enhance contrast with staining_contrast
    em_enhanced = np.clip(em * staining, 0, 1)

    # Gaussian blur FM (lower resolution) + sharpen EM
    psf_fm = gaussian_psf(11, 2.0)
    fm_blurred = fftconvolve(fm_shifted, psf_fm, mode='same')
    fm_blurred = np.clip(fm_blurred, 0, 1)

    psf_em = gaussian_psf(5, 0.8)
    em_sharp = fftconvolve(em_enhanced, psf_em, mode='same')
    em_sharp = np.clip(em_sharp, 0, 1)

    # Overlay: weighted average
    # FM weight = 0.4, EM weight = 0.6 (EM has more structural detail)
    overlay = 0.4 * fm_blurred + 0.6 * em_sharp
    y = normalize_01(overlay).astype(np.float32)

    # H_ideal: 3x3 affine registration matrix (identity + small shift)
    H_ideal = np.eye(3, dtype=np.float32)
    H_ideal[0, 2] = float(shift_y)
    H_ideal[1, 2] = float(shift_x)

    # reconstruction_baseline: y itself (overlay is the reconstruction)
    recon = y.copy()

    return y, H_ideal, {}, recon


def generate_clem():
    for tier, n_s, s_off, m_seed in [("public", 12, 0, 60000),
                                      ("dev", 20, 10000, 70000),
                                      ("hidden", 20, 20000, 80000)]:
        write_tier(
            "clem", tier, n_s, s_off, m_seed,
            CLEM_SPEC[tier],
            clem_forward,
            clem_phantom,
            "y = weighted_overlay(FM_blurred + EM_enhanced); registration error applied to FM; "
            "x_true: (2,256,256) [FM_channel, EM_channel]; H_ideal: (3,3) affine matrix",
            "y itself (registered overlay is the baseline reconstruction)",
        )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

MODALITY_GENERATORS = {
    "tirf": generate_tirf,
    "flim": generate_flim,
    "expansion": generate_expansion,
    "ism": generate_ism,
    "minflux": generate_minflux,
    "confocal_livecell": generate_confocal_livecell,
    "confocal_endomicroscopy": generate_confocal_endomicroscopy,
    "spinning_disk": generate_spinning_disk,
    "three_photon": generate_three_photon,
    "lattice_lightsheet": generate_lattice_lightsheet,
    "clem": generate_clem,
}


def main():
    # Allow selective generation via CLI args
    if len(sys.argv) > 1:
        modalities = sys.argv[1:]
    else:
        modalities = list(MODALITY_GENERATORS.keys())

    print("=" * 70)
    print("Optical/Fluorescence Microscopy Benchmark Dataset Generator")
    print(f"Output base: {BENCH_BASE}")
    print(f"Modalities: {modalities}")
    print("=" * 70)

    for mod in modalities:
        if mod not in MODALITY_GENERATORS:
            print(f"WARNING: Unknown modality '{mod}', skipping.")
            continue
        print(f"\n{'='*50}")
        print(f"Generating: {mod.upper()}")
        print(f"{'='*50}")
        try:
            MODALITY_GENERATORS[mod]()
        except Exception as e:
            import traceback
            print(f"ERROR generating {mod}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All modality datasets generated!")
    print("=" * 70)


if __name__ == "__main__":
    main()
