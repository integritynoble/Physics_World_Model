#!/usr/bin/env python3
"""Generate CBCT benchmark challenge datasets.

Pipeline per sample:
  1. Generate 3D procedural phantom (256^3) using simulate_phantoms.py
  2. Extract central axial slice as x_true (256x256)
  3. Radon-project the central slice to get sinogram_ideal
  4. Apply mismatch: beam hardening, scatter, noise, detector shift
  5. Package into HDF5 with generic schema (y, H_ideal, x_true)

Tiers:
  - public:  10 samples, dev recipes (seeds 100-109), mild mismatch
  - dev:     20 samples, dev recipes (seeds 8000-8019), medium mismatch
  - hidden:  20 samples, hidden recipes (seeds 9000-9019), severe mismatch

Geometry (from README):
  SID = 600 mm, SDD = 1200 mm, detector 512x512 @ 0.8mm pitch
  Volume: 256^3 @ 0.5mm voxel → 128mm FOV
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from simulate_phantoms import (
    generate_cbct_phantom,
    DEV_NVIEWS,
    HIDDEN_NVIEWS,
)

# ── Geometry ─────────────────────────────────────────────────────────────────
SID = 600.0       # mm, source to isocenter
SDD = 1200.0      # mm, source to detector
DET_PITCH = 0.8   # mm, detector pixel pitch
N_DET = 512        # detector pixels (we use central row for 2D sinogram)
VOX_SIZE = 0.5    # mm, isotropic voxel size
VOL_SIZE = 256    # voxels per axis

# Noise model
I0_PHOTONS = 5000  # incident photon count per ray (sparse-dose CBCT)
SIGMA_READOUT = 3.0  # detector readout noise (electrons)


def radon_project(image: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Radon transform of 2D image at given angles.

    Returns sinogram (n_views, n_det) where n_det = ceil(sqrt(2) * max_dim).
    Uses rotation-based projection matching skimage.transform.radon.
    """
    try:
        from skimage.transform import radon
        sino = radon(image, theta=angles_deg, circle=False)
        # radon returns (n_det, n_angles) → transpose to (n_angles, n_det)
        return sino.T
    except ImportError:
        pass

    # Fallback: manual rotation-based projection
    from scipy.ndimage import rotate
    H, W = image.shape
    diag = int(np.ceil(np.sqrt(H**2 + W**2)))
    pad_h = (diag - H) // 2
    pad_w = (diag - W) // 2
    padded = np.pad(image, ((pad_h, diag - H - pad_h), (pad_w, diag - W - pad_w)))

    sino = np.zeros((len(angles_deg), diag), dtype=np.float64)
    for i, theta in enumerate(angles_deg):
        rotated = rotate(padded, -theta, reshape=False, order=1)
        sino[i] = rotated.sum(axis=0)
    return sino


def apply_beam_hardening(sino: np.ndarray, beta: float) -> np.ndarray:
    """Simulate beam hardening: log-domain sinogram becomes nonlinear.

    Polychromatic beam: measured = -log(integral[S(E) exp(-mu(E)*L) dE])
    Approximated as: measured ≈ sino - beta * sino^2
    """
    if beta < 1e-6:
        return sino
    return sino - beta * sino**2


def apply_scatter(sino: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Add smooth scatter background to sinogram."""
    if fraction < 1e-6:
        return sino
    # Scatter is smooth, low-frequency signal proportional to total flux
    mean_signal = sino.mean()
    scatter_base = gaussian_filter(sino, sigma=[3.0, 8.0])
    scatter = fraction * mean_signal * (scatter_base / max(scatter_base.max(), 1e-8))
    return sino + scatter


def apply_detector_shift(sino: np.ndarray, shift_px: float) -> np.ndarray:
    """Shift sinogram along detector axis by fractional pixels."""
    if abs(shift_px) < 0.01:
        return sino
    from scipy.ndimage import shift as nd_shift
    return nd_shift(sino, [0, shift_px], order=1, mode='nearest')


def apply_noise(sino: np.ndarray, I0: float, sigma_readout: float,
                rng: np.random.Generator) -> np.ndarray:
    """Apply Poisson + Gaussian noise to log-domain sinogram.

    Model: measured = -log(Poisson(I0 * exp(-sino)) + N(0, sigma)) / I0_equiv
    """
    # Convert to transmission domain
    transmission = np.exp(-np.clip(sino, 0, 20))
    # Poisson photon counting
    counts = rng.poisson(I0 * transmission).astype(np.float64)
    # Readout noise
    counts += rng.normal(0, sigma_readout, counts.shape)
    counts = np.maximum(counts, 0.5)  # avoid log(0)
    # Back to log domain
    noisy = -np.log(counts / I0)
    return noisy


def generate_tier(
    tier: str,
    n_samples: int,
    base_seed: int,
    mode: str,
    nviews_list: list[int] | None,
    mismatch: dict,
    out_dir: Path,
):
    """Generate one tier of CBCT challenge data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"cbct_challenge_{tier}.h5"
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating CBCT {tier} tier: {n_samples} samples")
    print(f"  Mode: {mode}, base_seed: {base_seed}")
    print(f"  Mismatch: {mismatch}")
    print(f"  Output: {h5_path}")

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = f"PWM CBCT benchmark — {tier} tier (procedural phantoms, cone-beam geometry)"
        f.attrs["geometry"] = json.dumps({
            "SID_mm": SID, "SDD_mm": SDD, "det_pitch_mm": DET_PITCH,
            "n_det": N_DET, "vol_size": VOL_SIZE, "vox_size_mm": VOX_SIZE,
            "I0_photons": I0_PHOTONS, "sigma_readout": SIGMA_READOUT,
        })
        f.attrs["runner_type"] = "ct_fanbeam"
        f.attrs["tier"] = tier
        f.attrs["variant"] = "cbct"
        f.attrs["version"] = "1.0"

        spec_data = {}

        for i in range(n_samples):
            seed = base_seed + i
            rng = np.random.default_rng(seed + 50000)  # separate from phantom rng

            # Number of views
            if nviews_list and i < len(nviews_list):
                n_views = nviews_list[i]
            else:
                n_views = 256

            t0 = time.time()

            # 1. Generate 3D phantom
            try:
                mu_3d, recipe = generate_cbct_phantom(seed=seed, mode=mode, shape=(VOL_SIZE, VOL_SIZE, VOL_SIZE))
            except Exception as e:
                print(f"  [{i+1:2d}/{n_samples}] PHANTOM FAILED (seed={seed}): {e}")
                # Fallback: use simple Shepp-Logan-like phantom
                mu_3d = _fallback_phantom(seed, (VOL_SIZE, VOL_SIZE, VOL_SIZE))
                recipe = "fallback"

            t_phantom = time.time() - t0

            # 2. Extract central axial slice
            x_true = mu_3d[VOL_SIZE // 2].astype(np.float64)

            # 3. Radon projection
            angles_deg = np.linspace(0, 360 * (1 - 1/n_views), n_views, endpoint=False)
            sinogram_ideal = radon_project(x_true, angles_deg)

            # 4. Apply mismatch
            bh = mismatch["beam_hardening"] * rng.uniform(0.3, 1.0)
            sf = mismatch["scatter_fraction"] * rng.uniform(0.3, 1.0)
            ds = mismatch["detector_shift_u"] * rng.uniform(-1, 1)
            so_x = mismatch["source_offset_x"] * rng.uniform(-1, 1)
            dt = mismatch["detector_tilt"] * rng.uniform(-1, 1)

            sinogram = sinogram_ideal.copy()
            sinogram = apply_beam_hardening(sinogram, bh)
            sinogram = apply_scatter(sinogram, sf, rng)
            sinogram = apply_detector_shift(sinogram, ds)
            sinogram = apply_noise(sinogram, I0_PHOTONS, SIGMA_READOUT, rng)

            t_total = time.time() - t0

            # 5. Store in H5
            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("y", data=sinogram.astype(np.float32))
            grp.create_dataset("H_ideal", data=angles_deg.astype(np.float64))
            grp.create_dataset("x_true", data=x_true.astype(np.float32))
            grp.create_dataset("sinogram_ideal", data=sinogram_ideal.astype(np.float32))

            # Store per-sample spec
            spec_data[f"sample_{i:02d}"] = {
                "n_views": int(n_views),
                "recipe": recipe,
                "beam_hardening": round(float(bh), 4),
                "scatter_fraction": round(float(sf), 4),
                "detector_shift_u": round(float(ds), 4),
                "source_offset_x": round(float(so_x), 4),
                "detector_tilt": round(float(dt), 4),
            }

            # 6. Save preview images
            _save_preview(x_true, img_dir / f"sample_{i:02d}_x_true.png")
            _save_preview(sinogram, img_dir / f"sample_{i:02d}_y.png")
            _save_preview(sinogram_ideal, img_dir / f"sample_{i:02d}_sinogram_ideal.png")

            # Also save 3 orthogonal slices of the 3D phantom
            D = VOL_SIZE
            _save_preview(mu_3d[D//2], img_dir / f"sample_{i:02d}_axial.png")
            _save_preview(mu_3d[:, D//2, :], img_dir / f"sample_{i:02d}_coronal.png")
            _save_preview(mu_3d[:, :, D//2], img_dir / f"sample_{i:02d}_sagittal.png")

            print(f"  [{i+1:2d}/{n_samples}] {recipe:22s} views={n_views:3d} "
                  f"phantom={t_phantom:.0f}s total={t_total:.0f}s "
                  f"sino={sinogram.shape}")

    # Write per-sample true_spec
    spec_path = out_dir / "true_spec_samples.json"
    with open(spec_path, "w") as fp:
        json.dump(spec_data, fp, indent=2)

    size_mb = h5_path.stat().st_size / 1024 / 1024
    n_images = len(list(img_dir.glob("*.png")))
    print(f"\n  Done: {h5_path.name} ({size_mb:.1f} MB), {n_images} images")
    return h5_path


def _fallback_phantom(seed: int, shape: tuple) -> np.ndarray:
    """Simple ellipsoid phantom as fallback."""
    rng = np.random.default_rng(seed)
    D, H, W = shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)
    cz, cy, cx = D/2, H/2, W/2

    mu = np.zeros(shape, dtype=np.float32)
    # Outer ellipsoid (soft tissue)
    r = np.sqrt(((zz-cz)/(D*0.4))**2 + ((yy-cy)/(H*0.45))**2 + ((xx-cx)/(W*0.4))**2)
    mu[r < 1.0] = 0.25

    # Inner structures
    for _ in range(5):
        off = rng.uniform(-0.2, 0.2, 3) * np.array([D, H, W])
        radii = rng.uniform(0.05, 0.15, 3) * np.array([D, H, W])
        val = rng.uniform(0.1, 0.8)
        r2 = np.sqrt(((zz-cz-off[0])/max(radii[0],1))**2 +
                      ((yy-cy-off[1])/max(radii[1],1))**2 +
                      ((xx-cx-off[2])/max(radii[2],1))**2)
        mu[r2 < 1.0] = val

    return np.clip(mu, 0, 1)


def _save_preview(arr: np.ndarray, path: Path):
    """Save array as grayscale PNG."""
    from PIL import Image
    arr_f = arr.astype(np.float64)
    lo, hi = np.percentile(arr_f, [1, 99])
    if hi - lo > 1e-8:
        arr_f = np.clip((arr_f - lo) / (hi - lo), 0, 1)
    else:
        arr_f = np.clip(arr_f, 0, 1)
    img = Image.fromarray((arr_f * 255).astype(np.uint8), "L")
    img.save(path, format="PNG")


def main():
    base_dir = Path(__file__).parent

    # Mismatch parameters (from true_spec.json files)
    mismatch_public = {
        "source_offset_x": 0.80, "source_offset_z": 0.50,
        "detector_tilt": 0.15, "detector_shift_u": 1.20,
        "beam_hardening": 0.06, "scatter_fraction": 0.04,
    }
    mismatch_dev = {
        "source_offset_x": 0.50, "source_offset_z": 0.30,
        "detector_tilt": 0.10, "detector_shift_u": 0.80,
        "beam_hardening": 0.04, "scatter_fraction": 0.03,
    }
    mismatch_hidden = {
        "source_offset_x": 1.50, "source_offset_z": 1.00,
        "detector_tilt": 0.35, "detector_shift_u": 2.20,
        "beam_hardening": 0.12, "scatter_fraction": 0.08,
    }

    # Public: 10 samples, dev recipes with unique seeds, mild mismatch, 256 views
    public_nviews = [256] * 10
    generate_tier(
        tier="public", n_samples=10, base_seed=100, mode="dev",
        nviews_list=public_nviews, mismatch=mismatch_public,
        out_dir=base_dir / "public",
    )

    # Dev: 20 samples, dev recipes, medium mismatch, variable views
    generate_tier(
        tier="dev", n_samples=20, base_seed=8000, mode="dev",
        nviews_list=DEV_NVIEWS, mismatch=mismatch_dev,
        out_dir=base_dir / "dev",
    )

    # Hidden: 20 samples, hidden recipes, severe mismatch, sparser views
    generate_tier(
        tier="hidden", n_samples=20, base_seed=9000, mode="hidden",
        nviews_list=HIDDEN_NVIEWS, mismatch=mismatch_hidden,
        out_dir=base_dir / "hidden",
    )

    print(f"\n{'='*60}")
    print("CBCT dataset generation complete!")
    print(f"  Public:  {base_dir / 'public'}")
    print(f"  Dev:     {base_dir / 'dev'}")
    print(f"  Hidden:  {base_dir / 'hidden'}")


if __name__ == "__main__":
    main()
