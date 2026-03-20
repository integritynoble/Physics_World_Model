#!/usr/bin/env python3
"""
Generate widefield fluorescence microscopy benchmark dataset.

Physics:
  - Fluorescence widefield imaging
  - x_true: fluorophore density (cell-like phantoms), (256,256) float32, [0,1]
  - Forward: Gaussian PSF convolution (sigma 2-3px) + out-of-focus haze + Poisson noise + camera readout
  - y: (256,256) float32, noisy measurement
  - H_ideal: (256,256) float32, noiseless PSF-blurred image
  - reconstruction_baseline: simple Wiener deconvolution (~10 dB SNR assumption)
  - Mismatch: PSF sigma, background level, autofluorescence

Tiers:
  public:  12 samples
  dev:     20 samples
  hidden:  20 samples

Output:
  datasets/benchmark/widefield/{tier}/widefield_challenge_{tier}.h5
  datasets/benchmark/widefield/{tier}/spec.json
  datasets/benchmark/widefield/{tier}/true_spec.json   (all tiers)
  datasets/benchmark/widefield/{tier}/images/sample_NN/...
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "datasets" / "benchmark" / "widefield"

IMAGE_SIZE = 256
PIXEL_SIZE_UM = 0.100       # 100 nm/pixel
WAVELENGTH_EM_UM = 0.525    # emission wavelength
NA_NOMINAL = 1.25
N_IMMERSION = 1.515

# PSF sigma range per task spec: 2-3 pixels
PSF_SIGMA_NOMINAL = 2.5     # pixels (in [2,3])

# ── Mismatch spec ranges ────────────────────────────────────────────────────

SPEC = {
    "public": {
        "psf_sigma_px":       {"min": 2.0,  "max": 3.0,  "unit": "pixels"},
        "background_level":   {"min": 5.0,  "max": 20.0, "unit": "photons/pixel"},
        "autofluorescence":   {"min": 0.02, "max": 0.10, "unit": "fraction"},
        "noise_level":        {"min": 500,  "max": 2000, "unit": "peak photons"},
    },
    "dev": {
        "psf_sigma_px":       {"min": 1.8,  "max": 3.5,  "unit": "pixels"},
        "background_level":   {"min": 8.0,  "max": 35.0, "unit": "photons/pixel"},
        "autofluorescence":   {"min": 0.03, "max": 0.15, "unit": "fraction"},
        "noise_level":        {"min": 300,  "max": 2000, "unit": "peak photons"},
    },
    "hidden": {
        "psf_sigma_px":       {"min": 1.5,  "max": 4.0,  "unit": "pixels"},
        "background_level":   {"min": 10.0, "max": 50.0, "unit": "photons/pixel"},
        "autofluorescence":   {"min": 0.05, "max": 0.20, "unit": "fraction"},
        "noise_level":        {"min": 200,  "max": 1500, "unit": "peak photons"},
    },
}


# ── PSF helpers ──────────────────────────────────────────────────────────────

def make_gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """2D Gaussian PSF, normalised to sum=1."""
    if size % 2 == 0:
        size += 1
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    psf /= psf.sum()
    return psf


# ── Phantom generators ───────────────────────────────────────────────────────

def make_shepp_logan_cell_phantom(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Shepp-Logan style phantom with cell-like ellipsoidal regions."""
    x = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[:H, :W]
    n_cells = rng.integers(5, 15)
    for _ in range(n_cells):
        cy = rng.uniform(H * 0.1, H * 0.9)
        cx = rng.uniform(W * 0.1, W * 0.9)
        a = rng.uniform(8, 30)
        b = a * rng.uniform(0.5, 1.0)
        angle = rng.uniform(0, np.pi)
        intensity = rng.uniform(0.3, 1.0)
        dy = yy - cy
        dx = xx - cx
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        r2 = ((dy * cos_a + dx * sin_a) / a) ** 2 + ((-dy * sin_a + dx * cos_a) / b) ** 2
        mask = r2 < 1.0
        # Smooth interior with radial falloff + texture
        texture = rng.uniform(0, 1, (H, W))
        texture = gaussian_filter(texture, sigma=rng.uniform(2, 5))
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        x[mask] += intensity * (0.6 + 0.4 * texture[mask])
    return x


def make_actin_network_phantom(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Thin curved filament network (actin-like)."""
    x = np.zeros((H, W), dtype=np.float64)
    n_filaments = rng.integers(15, 40)
    for _ in range(n_filaments):
        n_pts = rng.integers(30, 100)
        cy = rng.uniform(0, H)
        cx = rng.uniform(0, W)
        angle = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(1.0, 3.0)
        thickness = rng.uniform(0.5, 1.5)
        intensity = rng.uniform(0.4, 1.0)
        pts_y = [cy]
        pts_x = [cx]
        for _ in range(n_pts - 1):
            angle += rng.normal(0, 0.15)
            ny = pts_y[-1] + speed * np.sin(angle)
            nx = pts_x[-1] + speed * np.cos(angle)
            pts_y.append(ny)
            pts_x.append(nx)
        for py, px in zip(pts_y, pts_x):
            iy, ix = int(round(py)), int(round(px))
            r = max(1, int(np.ceil(thickness * 2.5)))
            y0, y1 = max(0, iy - r), min(H, iy + r + 1)
            x0, x1 = max(0, ix - r), min(W, ix + r + 1)
            if y1 > y0 and x1 > x0:
                yyg = np.arange(y0, y1)[:, None].astype(np.float64)
                xxg = np.arange(x0, x1)[None, :].astype(np.float64)
                d2 = (yyg - py) ** 2 + (xxg - px) ** 2
                x[y0:y1, x0:x1] += intensity * np.exp(-d2 / (2 * thickness ** 2))
    return x


def make_mitochondria_phantom(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Tubular mitochondrial networks."""
    x = np.zeros((H, W), dtype=np.float64)
    n_cells = rng.integers(2, 5)
    for _ in range(n_cells):
        cell_cy = rng.uniform(H * 0.2, H * 0.8)
        cell_cx = rng.uniform(W * 0.2, W * 0.8)
        cell_r = rng.uniform(30, 60)
        n_tubules = rng.integers(20, 50)
        for _ in range(n_tubules):
            start_angle = rng.uniform(0, 2 * np.pi)
            start_r = rng.uniform(cell_r * 0.3, cell_r * 0.9)
            sy = cell_cy + start_r * np.sin(start_angle)
            sx = cell_cx + start_r * np.cos(start_angle)
            walk_angle = start_angle + rng.normal(0, 0.5)
            length = rng.uniform(10, 30)
            thickness = rng.uniform(0.6, 1.5)
            intensity = rng.uniform(0.4, 0.9)
            n_steps = int(length / 0.5)
            for step in range(n_steps):
                walk_angle += rng.normal(0, 0.08)
                py = sy + step * 0.5 * np.sin(walk_angle)
                px = sx + step * 0.5 * np.cos(walk_angle)
                if not (0 <= py < H and 0 <= px < W):
                    break
                iy, ix = int(round(py)), int(round(px))
                r = max(1, int(np.ceil(thickness * 2)))
                y0, y1 = max(0, iy - r), min(H, iy + r + 1)
                x0, x1 = max(0, ix - r), min(W, ix + r + 1)
                if y1 > y0 and x1 > x0:
                    yyg = np.arange(y0, y1)[:, None].astype(np.float64)
                    xxg = np.arange(x0, x1)[None, :].astype(np.float64)
                    d2 = (yyg - py) ** 2 + (xxg - px) ** 2
                    x[y0:y1, x0:x1] += intensity * np.exp(-d2 / (2 * thickness ** 2))
    return x


PHANTOM_FNS = [
    make_shepp_logan_cell_phantom,
    make_actin_network_phantom,
    make_mitochondria_phantom,
]


def make_phantom(idx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fn = PHANTOM_FNS[idx % len(PHANTOM_FNS)]
    x_raw = fn(IMAGE_SIZE, IMAGE_SIZE, rng)
    xmin, xmax = float(x_raw.min()), float(x_raw.max())
    if xmax - xmin > 1e-8:
        return ((x_raw - xmin) / (xmax - xmin)).astype(np.float32)
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)


# ── Forward model ────────────────────────────────────────────────────────────

def widefield_forward(
    x_true: np.ndarray,
    psf_sigma: float,
    background_level: float,
    autofluorescence: float,
    noise_level: float,
    rng: np.random.Generator,
    readout_noise_std: float = 3.0,
    oof_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        y       : (256,256) float32 noisy measurement
        H_ideal : (256,256) float32 noiseless blurred image (PSF*x + haze + bg)
    """
    H, W = x_true.shape

    # In-focus PSF
    psf = make_gaussian_psf(51, psf_sigma)

    # Out-of-focus haze PSF (5x broader)
    psf_oof = make_gaussian_psf(51, psf_sigma * 5.0)

    # Scale to photon counts
    x_photons = x_true.astype(np.float64) * noise_level

    # In-focus signal
    in_focus = fftconvolve(x_photons, psf, mode="same")

    # Out-of-focus haze
    oof_signal = fftconvolve(x_photons, psf_oof, mode="same") * oof_fraction

    # Spatially varying autofluorescence background
    auto = np.abs(gaussian_filter(rng.standard_normal((H, W)), sigma=25))
    auto = auto / (auto.max() + 1e-8) * autofluorescence * background_level * H * W / (H * W)
    # Uniform background component
    bg = background_level + auto

    # Noiseless ideal image
    ideal = np.maximum(in_focus + oof_signal + bg, 0.01)
    H_ideal = (ideal / (ideal.max() + 1e-8)).astype(np.float32)

    # Poisson shot noise
    y = rng.poisson(np.maximum(ideal, 0.01)).astype(np.float64)

    # Camera readout noise (Gaussian)
    y += rng.normal(0, readout_noise_std, (H, W))
    y = np.maximum(y, 0)

    # Normalise to [0, 1]
    y_max = float(y.max())
    if y_max > 1e-8:
        y = y / y_max
    return y.astype(np.float32), H_ideal


# ── Wiener deconvolution baseline ───────────────────────────────────────────

def wiener_deconvolve(
    y: np.ndarray,
    psf: np.ndarray,
    snr_db: float = 10.0,
) -> np.ndarray:
    """Simple Wiener deconvolution in the frequency domain.

    Args:
        y      : (H, W) noisy measurement
        psf    : (K, K) PSF kernel
        snr_db : assumed SNR in dB for Wiener regularisation

    Returns:
        recon  : (H, W) float32, normalised to [0, 1]
    """
    H, W = y.shape
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = 1.0 / snr_linear

    # Pad PSF to image size
    psf64 = psf.astype(np.float64)
    psf_pad = np.zeros((H, W), dtype=np.float64)
    kh, kw = psf64.shape
    # Place PSF in top-left corner (standard convention for FFT convolution)
    ph, pw = kh // 2, kw // 2
    psf_pad[:kh, :kw] = psf64
    # Roll so PSF centre is at (0,0)
    psf_pad = np.roll(psf_pad, (-ph, -pw), axis=(0, 1))

    Y = np.fft.rfft2(y.astype(np.float64))
    H_psf = np.fft.rfft2(psf_pad)
    H_conj = np.conj(H_psf)
    H_sq = np.abs(H_psf) ** 2

    # Wiener filter: W = H* / (|H|^2 + 1/SNR)
    W_filter = H_conj / (H_sq + noise_power)
    X_est = W_filter * Y
    recon = np.fft.irfft2(X_est, s=(H, W))

    # Clip and normalise to [0, 1]
    recon = np.clip(recon, 0, None)
    rmax = float(recon.max())
    if rmax > 1e-8:
        recon /= rmax
    return recon.astype(np.float32)


def baseline_reconstruct(y: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Wiener deconvolution with nominal PSF (sigma = 2.5 px)."""
    psf = make_gaussian_psf(51, PSF_SIGMA_NOMINAL)
    return wiener_deconvolve(y, psf, snr_db=snr_db)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return 100.0
    dr = float(gt.max() - gt.min())
    if dr < 1e-12:
        return 0.0
    return float(10.0 * np.log10(dr ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    g, r = gt.astype(np.float64), recon.astype(np.float64)
    dr = float(g.max() - g.min())
    if dr < 1e-12:
        return 0.0
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2
    mu_g, mu_r = g.mean(), r.mean()
    cov = float(np.mean((g - mu_g) * (r - mu_r)))
    return float(
        ((2 * mu_g * mu_r + c1) * (2 * cov + c2))
        / ((mu_g ** 2 + mu_r ** 2 + c1) * (g.var() + r.var() + c2))
    )


# ── PNG helpers ──────────────────────────────────────────────────────────────

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    if not HAS_PIL:
        return
    img = np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img, "L").save(str(path))


# ── Tier generator ───────────────────────────────────────────────────────────

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    result = {}
    for k, v in spec.items():
        val = float(rng.uniform(v["min"], v["max"]))
        if k in ("noise_level",):
            val = round(val)
        result[k] = val
    return result


def generate_tier(tier: str, n_samples: int, seed_offset: int, mismatch_seed: int) -> None:
    """Generate one tier."""
    spec_ranges = SPEC[tier]
    tier_dir = OUT_BASE / tier
    images_dir = tier_dir / "images"
    tier_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"widefield_challenge_{tier}.h5"
    rng = np.random.default_rng(mismatch_seed)
    true_specs: dict = {}
    psnrs, ssims = [], []

    print(f"\n[{tier}] Generating {n_samples} samples -> {h5_path}")

    with h5py.File(h5_path, "w") as f:
        f.attrs["modality"] = "widefield_fluorescence_microscopy"
        f.attrs["tier"] = tier
        f.attrs["n_samples"] = n_samples
        f.attrs["image_size"] = IMAGE_SIZE
        f.attrs["pixel_size_um"] = PIXEL_SIZE_UM
        f.attrs["psf_model"] = "gaussian"
        f.attrs["forward_model"] = (
            "y = Poisson(PSF(sigma)*x + oof_haze + autofluorescence_bg) + readout_noise; "
            "H_ideal = noiseless blurred image (PSF*x + haze + bg), normalised to [0,1]"
        )
        f.attrs["baseline_method"] = "Wiener deconvolution, SNR assumption 10 dB"
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)

        for idx in range(n_samples):
            key = f"sample_{idx:02d}"
            phantom_rng_seed = seed_offset + idx
            x_true = make_phantom(idx, phantom_rng_seed)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "phantom_seed": phantom_rng_seed,
                               "phantom_type": ["shepp_logan_cell", "actin_network",
                                                "mitochondria"][idx % 3]}

            sample_rng = np.random.default_rng(mismatch_seed + idx + 1)
            y, H_ideal = widefield_forward(
                x_true,
                psf_sigma=mis["psf_sigma_px"],
                background_level=mis["background_level"],
                autofluorescence=mis["autofluorescence"],
                noise_level=mis["noise_level"],
                rng=sample_rng,
            )

            recon = baseline_reconstruct(y, snr_db=10.0)

            psnr = compute_psnr(x_true, recon)
            ssim = compute_ssim(x_true, recon)
            psnrs.append(psnr)
            ssims.append(ssim)

            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true, compression="gzip")
            grp.create_dataset("y", data=y, compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")

            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["metadata"] = json.dumps({
                "scene": true_specs[key]["phantom_type"],
                "shapes": {
                    "x_true": list(x_true.shape),
                    "y": list(y.shape),
                    "H_ideal": list(H_ideal.shape),
                    "reconstruction_baseline": list(recon.shape),
                },
                "x_true_range": [float(x_true.min()), float(x_true.max())],
                "y_range": [float(y.min()), float(y.max())],
                "H_ideal_range": [float(H_ideal.min()), float(H_ideal.max())],
                "recon_range": [float(recon.min()), float(recon.max())],
                "psnr_baseline_db": round(psnr, 2),
                "ssim_baseline": round(ssim, 4),
            })

            # Images
            sample_img_dir = images_dir / f"sample_{idx:02d}"
            sample_img_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_img_dir / "x_true.png")
            _save_png(y, sample_img_dir / "y_measurement.png")
            _save_png(H_ideal, sample_img_dir / "H_ideal.png")
            _save_png(recon, sample_img_dir / "reconstruction.png")

            with open(sample_img_dir / "spec.json", "w") as sf:
                json.dump({"true_spec": mis, "spec_ranges": spec_ranges,
                           "psnr_db": psnr, "ssim": ssim}, sf, indent=2)

            # Print shapes and ranges on first sample and every 5 after
            if idx == 0 or (idx + 1) % 5 == 0:
                print(f"  {key}: x_true {x_true.shape} [{x_true.min():.4f},{x_true.max():.4f}] "
                      f"y {y.shape} [{y.min():.4f},{y.max():.4f}] "
                      f"H_ideal {H_ideal.shape} [{H_ideal.min():.4f},{H_ideal.max():.4f}] "
                      f"recon {recon.shape} [{recon.min():.4f},{recon.max():.4f}] "
                      f"PSF sigma={mis['psf_sigma_px']:.2f}px  "
                      f"bg={mis['background_level']:.1f}  "
                      f"autofl={mis['autofluorescence']:.3f}  "
                      f"PSNR={psnr:.2f}dB  SSIM={ssim:.4f}")

        f.attrs["mean_psnr_baseline_db"] = float(np.mean(psnrs))
        f.attrs["mean_ssim_baseline"] = float(np.mean(ssims))

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    print(f"  [{tier}] Done: {n_samples} samples | "
          f"mean PSNR={np.mean(psnrs):.2f} dB | mean SSIM={np.mean(ssims):.4f}")
    print(f"  [{tier}] HDF5: {h5_path}  "
          f"({os.path.getsize(h5_path) / 1e6:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Widefield Fluorescence Microscopy Benchmark Dataset Generator")
    print(f"Output: {OUT_BASE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"PSF model: Gaussian, sigma in [2.0, 3.0] px (nominal={PSF_SIGMA_NOMINAL}px)")
    print(f"Baseline: Wiener deconvolution (SNR ~10 dB)")
    print("=" * 70)

    generate_tier("public",  n_samples=12, seed_offset=0,     mismatch_seed=1000)
    generate_tier("dev",     n_samples=20, seed_offset=10000, mismatch_seed=11000)
    generate_tier("hidden",  n_samples=20, seed_offset=20000, mismatch_seed=21000)

    print("\n" + "=" * 70)
    print("Widefield benchmark generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
