"""Build multi-coil MRI benchmark dataset (PWM-style).

Forward model (parallel imaging with B0 mismatch):
    y_c = mask · F( S_c_true · x_warped · exp(i·2π·B0_hz·TE·b0_map) ) + n_c

where:
  S_c_true = S_c_nominal · (1 + ε_c)          [coil sensitivity mismatch]
  x_warped  = warp(x_true, δr)                  [gradient nonlinearity mismatch]
  b0_map    = smooth B0 field                    [field inhomogeneity mismatch]
  trajectory error applied as per-line k-ramp    [k-trajectory mismatch]

Nominal operator (what algorithms see):
  H_nominal: y_c = mask · F( S_c_nominal · x )

Run from the mri/ directory:
    python build_dataset.py

Creates:
  public/   mri_challenge_public.h5  + images/  (11 Shepp-Logan, mild mismatch)
  dev/      mri_challenge_dev.h5     + images/  (20 brain-like, mild mismatch)
  hidden/   mri_challenge_hidden.h5  + images/  (20 adversarial, severe mismatch)
"""

from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_scenes import generate_mri_gt, shepp_logan_phantom


# ── Constants ──────────────────────────────────────────────────────────────────

SHAPE    = (256, 256)
N_COILS  = 8
ACCEL    = 4
ACS_FRAC = 0.08
TE_S     = 0.025   # 25 ms echo time

SPEC_RANGES = {
    "public": {
        "B0_inhomog_hz":         {"min":  5.0,  "max": 15.0,  "unit": "Hz"},
        "gradient_nonlin_frac":  {"min":  0.001,"max":  0.003,"unit": "frac"},
        "coil_sensitivity_frac": {"min":  0.01, "max":  0.03, "unit": "frac"},
        "k_trajectory_frac":     {"min":  0.001,"max":  0.003,"unit": "frac"},
        "noise_sigma":           {"min":  0.01, "max":  0.02, "unit": "rel"},
    },
    "dev": {
        "B0_inhomog_hz":         {"min":  5.0,  "max": 20.0,  "unit": "Hz"},
        "gradient_nonlin_frac":  {"min":  0.001,"max":  0.005,"unit": "frac"},
        "coil_sensitivity_frac": {"min":  0.01, "max":  0.05, "unit": "frac"},
        "k_trajectory_frac":     {"min":  0.001,"max":  0.005,"unit": "frac"},
        "noise_sigma":           {"min":  0.01, "max":  0.03, "unit": "rel"},
    },
    "hidden": {
        "B0_inhomog_hz":         {"min": 20.0,  "max": 60.0,  "unit": "Hz"},
        "gradient_nonlin_frac":  {"min":  0.005,"max":  0.02, "unit": "frac"},
        "coil_sensitivity_frac": {"min":  0.05, "max":  0.15, "unit": "frac"},
        "k_trajectory_frac":     {"min":  0.005,"max":  0.02, "unit": "frac"},
        "noise_sigma":           {"min":  0.03, "max":  0.06, "unit": "rel"},
    },
}


# ── Coil sensitivity maps ──────────────────────────────────────────────────────

def generate_coil_maps(shape, n_coils, rng, coil_radius_frac=0.58):
    """N coils on a ring: Gaussian magnitude + smooth spatially varying phase."""
    H, W = shape
    yy = np.linspace(-0.5, 0.5, H, dtype=np.float32)[:, None]
    xx = np.linspace(-0.5, 0.5, W, dtype=np.float32)[None, :]
    coil_maps = np.zeros((n_coils, H, W), dtype=np.complex64)
    sigma = float(rng.uniform(0.22, 0.36))
    for c in range(n_coils):
        angle = 2.0 * np.pi * c / n_coils + float(rng.uniform(-0.1, 0.1))
        cy = coil_radius_frac * np.sin(angle)
        cx = coil_radius_frac * np.cos(angle)
        mag = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
        phase_n = rng.standard_normal((H, W)).astype(np.float32)
        phase_n = gaussian_filter(phase_n, sigma=float(rng.uniform(8.0, 20.0)))
        phase_scale = float(rng.uniform(0.1, 0.4)) * np.pi
        phase = phase_n / (float(np.abs(phase_n).max()) + 1e-6) * phase_scale
        coil_maps[c] = (mag.astype(np.float32) * np.exp(1j * phase)).astype(np.complex64)
    return coil_maps


def generate_coil_perturbation(coil_maps, strength_frac, rng):
    """Smooth complex ε_c per coil; S_true = S_nominal * (1 + ε_c)."""
    n_coils, H, W = coil_maps.shape
    perturb = np.zeros((n_coils, H, W), dtype=np.complex64)
    for c in range(n_coils):
        re = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                             sigma=float(rng.uniform(5.0, 20.0)))
        im = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                             sigma=float(rng.uniform(5.0, 20.0)))
        p = (re + 1j * im).astype(np.complex64)
        perturb[c] = strength_frac * p / (float(np.abs(p).max()) + 1e-8)
    return perturb


# ── B0 field map ───────────────────────────────────────────────────────────────

def generate_b0_map(shape, rng):
    """Smooth field map normalised to [-1, 1]."""
    H, W = shape
    yy = np.linspace(-0.5, 0.5, H, dtype=np.float32)[:, None]
    xx = np.linspace(-0.5, 0.5, W, dtype=np.float32)[None, :]
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    grad = (np.cos(angle) * yy + np.sin(angle) * xx).astype(np.float32)
    noise = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                            sigma=float(rng.uniform(10.0, 35.0)))
    noise -= noise.mean()
    noise /= max(float(np.abs(noise).max()), 1e-6)
    b0 = 0.55 * grad + 0.45 * noise
    b0 /= max(float(np.abs(b0).max()), 1e-6)
    return b0.astype(np.float32)


# ── Gradient nonlinearity (geometric warp) ─────────────────────────────────────

def generate_warp_field(shape, strength_frac, rng):
    """Smooth displacement (dy, dx) in pixels; max = strength_frac * min(H, W)."""
    H, W = shape
    max_d = strength_frac * min(H, W)
    fields = []
    for _ in range(2):
        f = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                            sigma=float(rng.uniform(20.0, 50.0)))
        f /= max(float(np.abs(f).max()), 1e-6)
        fields.append(f * max_d)
    return np.stack(fields, axis=0).astype(np.float32)  # (2, H, W)


def apply_warp(x, warp):
    """Warp image by displacement field; handles complex arrays."""
    H, W = x.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    coords = [np.clip(yy + warp[0], 0, H - 1), np.clip(xx + warp[1], 0, W - 1)]
    if np.iscomplexobj(x):
        re = map_coordinates(x.real.astype(np.float64), coords, order=1, mode='reflect')
        im = map_coordinates(x.imag.astype(np.float64), coords, order=1, mode='reflect')
        return (re + 1j * im).astype(np.complex64)
    return map_coordinates(x.astype(np.float64), coords, order=1, mode='reflect').astype(np.float32)


# ── k-trajectory error ─────────────────────────────────────────────────────────

def apply_k_trajectory_error(kspace_c, mask_1d, strength_frac, rng):
    """Per-line fractional k-space shift via phase ramp (gradient timing error)."""
    H, W = kspace_c.shape
    out = kspace_c.copy()
    sampled = np.where(mask_1d)[0]
    shifts = rng.uniform(-strength_frac * W, strength_frac * W, size=len(sampled))
    kx = np.arange(W, dtype=np.float32)
    for i, ky in enumerate(sampled):
        ramp = np.exp(1j * 2.0 * np.pi * shifts[i] * kx / W).astype(np.complex64)
        out[ky] = kspace_c[ky] * ramp
    return out


# ── Undersampling mask ─────────────────────────────────────────────────────────

def generate_vds_mask(n_lines, accel=4, acs_frac=0.08, seed=None):
    """Variable-density Cartesian ky mask, returns bool (n_lines,)."""
    rng = np.random.default_rng(seed)
    n_acs = max(8, int(n_lines * acs_frac))
    n_total = max(n_acs, n_lines // accel)
    n_outer = n_total - n_acs
    mask = np.zeros(n_lines, dtype=bool)
    start = (n_lines - n_acs) // 2
    mask[start:start + n_acs] = True
    outer = np.where(~mask)[0]
    probs = np.exp(-((outer - n_lines // 2) ** 2) / (2.0 * (n_lines * 0.25) ** 2))
    probs /= probs.sum()
    chosen = rng.choice(outer, size=min(n_outer, len(outer)), replace=False, p=probs)
    mask[chosen] = True
    return mask


# ── Multi-coil MRI forward model ──────────────────────────────────────────────

def mri_forward_multicoil(x_true, coil_maps_nominal, coil_perturb, mask_1d,
                           b0_hz, b0_map, warp_field, k_traj_frac, noise_sigma, rng):
    """True (mismatched) multi-coil MRI forward model.

    Nominal model algorithms assume: y_c = mask · F(S_c · x)
    True acquisition:
      x_warped = warp(x_true, δr)                      [gradient nonlin]
      x_mod    = x_warped · exp(i·2π·B0·TE·b0_map)     [B0 mismatch]
      S_c_true = S_c_nominal · (1 + ε_c)               [coil mismatch]
      y_c      = mask · k_traj_error(F(S_c_true·x_mod)) + n_c

    Returns y: (C, H, W) complex64
    """
    C, H, W = coil_maps_nominal.shape
    mask_2d = mask_1d[:, np.newaxis] * np.ones((1, W), dtype=bool)

    x_warped = apply_warp(x_true, warp_field)
    phi = (2.0 * np.pi * b0_hz * TE_S * b0_map).astype(np.float32)
    x_mod = x_warped.astype(np.complex64) * np.exp(1j * phi).astype(np.complex64)

    coil_maps_true = coil_maps_nominal * (1.0 + coil_perturb)

    y_multi = np.zeros((C, H, W), dtype=np.complex64)
    for c in range(C):
        kspace_c = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x_mod * coil_maps_true[c])))
        kspace_m = kspace_c * mask_2d
        kspace_m = apply_k_trajectory_error(kspace_m, mask_1d, k_traj_frac, rng)
        sig_std = float(np.abs(kspace_c[mask_2d]).std()) + 1e-8
        noise = ((rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W)))
                 * noise_sigma * sig_std).astype(np.complex64)
        y_multi[c] = kspace_m + noise * mask_2d

    return y_multi.astype(np.complex64)


def rss_recon(y_kspace):
    """Zero-filled RSS reconstruction from undersampled multi-coil k-space."""
    imgs = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(y_kspace, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )
    rss = np.sqrt(np.sum(np.abs(imgs) ** 2, axis=0)).astype(np.float32)
    if rss.max() > 1e-6:
        rss /= rss.max()
    return rss


# ── Image helpers ──────────────────────────────────────────────────────────────

def _norm(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)

def _save_png(arr, path):
    Image.fromarray(np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L").save(path)

def _resize(arr, h, w):
    pil = Image.fromarray(np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L")
    return np.array(pil.resize((w, h), Image.LANCZOS)) / 255.0


def make_sample_images(x_true, y_kspace, coil_maps, mask_1d, b0_map, sample_dir, spec):
    os.makedirs(sample_dir, exist_ok=True)
    C, H, W = y_kspace.shape

    rss = rss_recon(y_kspace)
    kspace_log = _norm(np.log1p(np.mean(np.abs(y_kspace), axis=0)))
    mask_2d = (mask_1d[:, None] * np.ones((1, W))).astype(np.float32)

    _save_png(x_true,    os.path.join(sample_dir, "ground_truth.png"))
    _save_png(rss,       os.path.join(sample_dir, "rss_reconstruction.png"))
    _save_png(kspace_log,os.path.join(sample_dir, "kspace_magnitude.png"))
    _save_png(mask_2d,   os.path.join(sample_dir, "undersampling_mask.png"))
    _save_png(_norm(b0_map), os.path.join(sample_dir, "b0_map.png"))

    # Coil sensitivity mosaic (2×4)
    cols = 4
    rows = (C + cols - 1) // cols
    th, tw = 64, 64
    mosaic = np.zeros((rows * th, cols * tw), dtype=np.float32)
    for c in range(C):
        r, col = c // cols, c % cols
        mosaic[r*th:(r+1)*th, col*tw:(col+1)*tw] = _resize(np.abs(coil_maps[c]), th, tw)
    _save_png(mosaic, os.path.join(sample_dir, "coil_sensitivity.png"))

    # 2×3 overview
    th2, tw2 = 128, 128
    ov = np.zeros((2 * th2, 3 * tw2), dtype=np.float32)
    ov[0:th2,  0:tw2]      = _resize(x_true,     th2, tw2)
    ov[0:th2,  tw2:2*tw2]  = _resize(rss,        th2, tw2)
    ov[0:th2,  2*tw2:]     = _resize(kspace_log, th2, tw2)
    ov[th2:,   0:tw2]      = _resize(mask_2d,    th2, tw2)
    ov[th2:,   tw2:2*tw2]  = _resize(_norm(b0_map), th2, tw2)
    ov[th2:,   2*tw2:]     = _resize(_norm(np.abs(coil_maps).mean(0)), th2, tw2)
    _save_png(ov, os.path.join(sample_dir, "overview.png"))

    with open(os.path.join(sample_dir, "spec.json"), "w") as fh:
        json.dump(spec, fh, indent=2)


# ── Tier builder ───────────────────────────────────────────────────────────────

def build_tier(tier, scenes, output_dir, spec_ranges_key, base_seed):
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f"mri_challenge_{tier}.h5")
    images_dir = os.path.join(output_dir, "images")
    sr = SPEC_RANGES[spec_ranges_key]
    rng = np.random.default_rng(base_seed)
    table = []

    with h5py.File(h5_path, "w") as hf:
        for i, (scene_name, x_true, recipe) in enumerate(scenes):
            grp = hf.create_group(f"sample_{i:02d}")

            def _u(k):
                return float(rng.uniform(sr[k]["min"], sr[k]["max"]))

            b0_hz       = _u("B0_inhomog_hz")
            grad_frac   = _u("gradient_nonlin_frac")
            coil_frac   = _u("coil_sensitivity_frac")
            ktraj_frac  = _u("k_trajectory_frac")
            noise_sigma = _u("noise_sigma")

            coil_maps   = generate_coil_maps(SHAPE, N_COILS, rng)
            b0_map      = generate_b0_map(SHAPE, rng)
            warp_field  = generate_warp_field(SHAPE, grad_frac, rng)
            coil_perturb = generate_coil_perturbation(coil_maps, coil_frac, rng)
            mask_1d     = generate_vds_mask(SHAPE[0], ACCEL, ACS_FRAC,
                                            seed=base_seed + i * 997 + 3)

            y_kspace = mri_forward_multicoil(
                x_true, coil_maps, coil_perturb, mask_1d,
                b0_hz, b0_map, warp_field, ktraj_frac, noise_sigma, rng,
            )

            grp.create_dataset("x_true",     data=x_true,                 compression="gzip")
            grp.create_dataset("y_kspace",   data=y_kspace,               compression="gzip")
            grp.create_dataset("mask",       data=mask_1d.astype(np.uint8))
            grp.create_dataset("coil_maps",  data=coil_maps,              compression="gzip")
            grp.create_dataset("B0_map",     data=b0_map,                 compression="gzip")
            grp.create_dataset("warp_field", data=warp_field,             compression="gzip")

            true_spec = {
                "B0_inhomog_hz":         round(b0_hz, 6),
                "gradient_nonlin_frac":  round(grad_frac, 6),
                "coil_sensitivity_frac": round(coil_frac, 6),
                "k_trajectory_frac":     round(ktraj_frac, 6),
                "noise_sigma":           round(noise_sigma, 6),
            }
            metadata = {
                "scene": scene_name, "shape": list(SHAPE), "n_coils": N_COILS,
                "accel_factor": ACCEL, "te_s": TE_S, "recipe": recipe,
                "n_sampled_lines": int(mask_1d.sum()),
            }
            grp.attrs["metadata"]    = json.dumps(metadata)
            grp.attrs["spec_ranges"] = json.dumps(sr)
            grp.attrs["true_spec"]   = json.dumps(true_spec)

            sample_dir = os.path.join(images_dir, f"sample_{i:02d}_{scene_name}")
            make_sample_images(x_true, y_kspace, coil_maps, mask_1d, b0_map,
                               sample_dir,
                               {"scene": scene_name, "spec_ranges": sr, "true_spec": true_spec})

            row = {**true_spec, "sample_idx": i, "scene": scene_name,
                   "recipe": recipe, "n_sampled_lines": int(mask_1d.sum())}
            table.append(row)
            print(f"  [{tier}] {i:02d} {scene_name}: "
                  f"B0={b0_hz:.1f}Hz grad={grad_frac:.4f} coil={coil_frac:.3f} "
                  f"ktraj={ktraj_frac:.4f} σ={noise_sigma:.4f} recipe={recipe}")
    return table


# ── README writer ──────────────────────────────────────────────────────────────

def write_tier_readme(tier, output_dir, table, spec_ranges_key):
    sr = SPEC_RANGES[spec_ranges_key]
    rows = "".join(
        f"| sample_{s['sample_idx']:02d}  | {s['scene']:<22} | "
        f"{s['B0_inhomog_hz']:6.1f} | {s['gradient_nonlin_frac']:.4f} | "
        f"{s['coil_sensitivity_frac']:.3f} | {s['k_trajectory_frac']:.4f} | "
        f"{s['noise_sigma']:.4f} | {s['recipe']} |\n"
        for s in table
    )
    source = {
        "public": "Shepp-Logan phantom variants (11 analytic samples)",
        "dev":    "Procedural brain-like phantoms (20 samples, natural tissue statistics)",
        "hidden": "Adversarial stress-test phantoms (20 samples, severe mismatch)",
    }[tier]
    text = f"""# MRI {tier.capitalize()} Tier

## Source
{source}

## Per-Sample Mismatch Values

| Sample     | Scene                  | B0 (Hz) | grad_nonlin | coil_sens | k_traj | noise_σ | recipe |
|------------|------------------------|---------|-------------|-----------|--------|---------|--------|
{rows}
## HDF5 Datasets (per sample)

| Key           | Shape                        | Dtype     | Description                          |
|---------------|------------------------------|-----------|--------------------------------------|
| `x_true`      | (256, 256)                   | float32   | GT magnitude image [0, 1]            |
| `y_kspace`    | ({N_COILS}, 256, 256)                | complex64 | Undersampled k-space per coil        |
| `mask`        | (256,)                       | uint8     | 1D ky undersampling mask             |
| `coil_maps`   | ({N_COILS}, 256, 256)                | complex64 | **Nominal** coil sensitivity maps    |
| `B0_map`      | (256, 256)                   | float32   | True B0 field map (oracle)           |
| `warp_field`  | (2, 256, 256)                | float32   | True gradient warp (dy, dx) px       |

## Image Files (per sample)

- `ground_truth.png`       — True MR magnitude image
- `rss_reconstruction.png` — Zero-filled RSS (shows aliasing artefacts)
- `kspace_magnitude.png`   — Log|y| averaged over coils
- `undersampling_mask.png` — Cartesian ky undersampling pattern
- `coil_sensitivity.png`   — Mosaic of |S_c| for all {N_COILS} coils
- `b0_map.png`             — B0 field inhomogeneity map
- `overview.png`           — 2×3 summary grid
- `spec.json`              — Per-sample mismatch specification
"""
    with open(os.path.join(output_dir, "README.md"), "w") as fh:
        fh.write(text)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 70)
    print("Multi-coil MRI benchmark (PWM parallel imaging)")
    print(f"Shape={SHAPE}  Coils={N_COILS}  Accel={ACCEL}x  TE={TE_S*1e3:.0f}ms")
    print("Mismatch: B0_inhomog · gradient_nonlin · coil_sensitivity · k_trajectory")
    print("=" * 70)

    print("\n[public] Shepp-Logan variants (11 samples)...")
    pub = [(f"shepp_logan_{i:02d}", shepp_logan_phantom(SHAPE, i), "shepp_logan")
           for i in range(11)]
    pub_dir = os.path.join(base_dir, "public")
    pub_t = build_tier("public", pub, pub_dir, "public", base_seed=1000)
    write_tier_readme("public", pub_dir, pub_t, "public")

    print("\n[dev] Brain-like procedural (20 samples)...")
    dev = [(f"proc_dev_{i:02d}", *generate_mri_gt(5000+i, "dev", SHAPE)) for i in range(20)]
    dev_dir = os.path.join(base_dir, "dev")
    dev_t = build_tier("dev", dev, dev_dir, "dev", base_seed=2000)
    write_tier_readme("dev", dev_dir, dev_t, "dev")

    print("\n[hidden] Adversarial (20 samples)...")
    hid = [(f"proc_hidden_{i:02d}", *generate_mri_gt(8000+i, "hidden", SHAPE)) for i in range(20)]
    hid_dir = os.path.join(base_dir, "hidden")
    hid_t = build_tier("hidden", hid, hid_dir, "hidden", base_seed=3000)
    write_tier_readme("hidden", hid_dir, hid_t, "hidden")

    print("\n" + "=" * 70)
    print(f"Done.  public={len(pub_t)}  dev={len(dev_t)}  hidden={len(hid_t)} samples")
    print("=" * 70)


if __name__ == "__main__":
    main()
