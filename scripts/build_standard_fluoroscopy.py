"""
build_standard_fluoroscopy.py
Build standard fluoroscopy (X-ray video frame) dataset for PWM benchmark.

x_true : (384,288) float32 [0,1] — 2-D X-ray attenuation image
y_ideal : (384,288) float32 [0,1] — fluoroscopic measurement (Beer-Lambert + Gaussian blur)

Source / citation:
  CVC-ClinicDB, Bernal et al. CMIG 2015,
  doi:10.1016/j.compmedimag.2015.02.007
  Synthetic X-ray phantom used (CVC-ClinicDB may require registration).
"""

import sys
import json
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "datasets" / "benchmark" / "fluoroscopy" / "standard"
MODALITY = "fluoroscopy"
N_SAMPLES = 10
IMG_H = 384
IMG_W = 288


# ---------------------------------------------------------------------------
# Synthetic X-ray attenuation phantom
# ---------------------------------------------------------------------------

def _gaussian_2d(H, W, cy, cx, sigma_y, sigma_x, amplitude):
    """Return 2-D Gaussian blob."""
    y_arr = np.arange(H)
    x_arr = np.arange(W)
    Y, X = np.meshgrid(y_arr, x_arr, indexing="ij")
    blob = amplitude * np.exp(
        -0.5 * (((Y - cy) / sigma_y) ** 2 + ((X - cx) / sigma_x) ** 2)
    )
    return blob.astype(np.float32)


def _make_xray_phantom(rng):
    """Return (IMG_H, IMG_W) float32 attenuation image [0,1].

    Anatomical model: chest/abdomen with:
    - Soft-tissue background (uniform low attenuation)
    - Ribcage (curved elliptical arcs)
    - Spine (central vertical high-attenuation column)
    - Random circular bone-like high-attenuation inclusions
    - Soft-tissue organs (lower attenuation blobs)
    """
    phantom = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    cy_centre = IMG_H / 2
    cx_centre = IMG_W / 2

    # Body outline (soft tissue fill — ellipse)
    Y, X = np.ogrid[:IMG_H, :IMG_W]
    body_ry = IMG_H * 0.42
    body_rx = IMG_W * 0.43
    body_mask = ((Y - cy_centre) / body_ry) ** 2 + ((X - cx_centre) / body_rx) ** 2 <= 1
    phantom[body_mask] = rng.uniform(0.10, 0.20)

    # Lungs (lower attenuation ellipses, bilateral)
    for side in [-1, 1]:
        lung_cx = cx_centre + side * IMG_W * 0.18
        lung_cy = cy_centre - IMG_H * 0.05
        lung_ry = IMG_H * 0.28
        lung_rx = IMG_W * 0.14
        lung_mask = ((Y - lung_cy) / lung_ry) ** 2 + ((X - lung_cx) / lung_rx) ** 2 <= 1
        phantom[lung_mask] = rng.uniform(0.02, 0.06)

    # Spine (central vertical column, high attenuation)
    spine_cx = cx_centre
    spine_width = IMG_W * 0.06
    spine_mask = np.abs(X - spine_cx) < spine_width
    phantom[spine_mask & body_mask] = rng.uniform(0.55, 0.75)

    # Ribs (curved elliptical arcs, bilateral, multiple levels)
    n_ribs = rng.integers(6, 10)
    for i in range(n_ribs):
        rib_cy = cy_centre - IMG_H * 0.25 + i * IMG_H * 0.06
        for side in [-1, 1]:
            rib_cx = cx_centre + side * IMG_W * 0.22
            rib_ry = rng.uniform(IMG_H * 0.04, IMG_H * 0.07)
            rib_rx = rng.uniform(IMG_W * 0.06, IMG_W * 0.10)
            rib_thickness = rng.uniform(3, 7)
            dist = np.sqrt(((Y - rib_cy) / rib_ry) ** 2 + ((X - rib_cx) / rib_rx) ** 2)
            rib_mask = (dist >= 0.85) & (dist <= 1.0) & body_mask
            phantom[rib_mask] = rng.uniform(0.50, 0.72)

    # Random circular bone-like high-attenuation objects (vertebral bodies, surgical clips etc.)
    n_bones = rng.integers(4, 9)
    for _ in range(n_bones):
        bcy = rng.integers(int(IMG_H * 0.15), int(IMG_H * 0.85))
        bcx = rng.integers(int(IMG_W * 0.10), int(IMG_W * 0.90))
        br = rng.uniform(4, 14)
        bval = rng.uniform(0.45, 0.85)
        bmask = (Y - bcy) ** 2 + (X - bcx) ** 2 <= br ** 2
        phantom[bmask & body_mask] = bval

    # Soft-tissue organ blobs (liver, heart, diaphragm)
    n_organs = rng.integers(3, 6)
    for _ in range(n_organs):
        ocy = rng.uniform(IMG_H * 0.2, IMG_H * 0.8)
        ocx = rng.uniform(IMG_W * 0.15, IMG_W * 0.85)
        sigma_y = rng.uniform(IMG_H * 0.05, IMG_H * 0.18)
        sigma_x = rng.uniform(IMG_W * 0.05, IMG_W * 0.15)
        amp = rng.uniform(0.06, 0.25)
        blob = _gaussian_2d(IMG_H, IMG_W, ocy, ocx, sigma_y, sigma_x, amp)
        phantom += blob * body_mask.astype(np.float32)

    # Clip and normalise
    phantom = np.clip(phantom, 0, None)
    pmax = phantom.max()
    if pmax > 0:
        phantom /= pmax
    return phantom.astype(np.float32)


# ---------------------------------------------------------------------------
# Beer-Lambert fluoroscopic forward model
# ---------------------------------------------------------------------------

def _gaussian_blur(image, sigma=2.0):
    """Apply approximate Gaussian blur via separable 1-D convolution."""
    ksize = int(6 * sigma + 1) | 1  # ensure odd
    half = ksize // 2
    k1d = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2).astype(np.float32)
    k1d /= k1d.sum()
    # Blur along rows then columns
    out = np.apply_along_axis(lambda row: np.convolve(row, k1d, mode="same"), 1, image)
    out = np.apply_along_axis(lambda col: np.convolve(col, k1d, mode="same"), 0, out)
    return out.astype(np.float32)


def _fluoroscopy_forward(x_true, mu=3.0, sigma_blur=1.5):
    """Beer-Lambert transmission: y = exp(-mu * x_true), then Gaussian blur.

    mu: effective linear attenuation coefficient scaling (dimensionless here).
    """
    transmission = np.exp(-mu * x_true).astype(np.float32)
    y = _gaussian_blur(transmission, sigma=sigma_blur)
    # Normalise to [0,1]
    ymin, ymax = y.min(), y.max()
    if ymax > ymin:
        y = (y - ymin) / (ymax - ymin)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fluoroscopy] Output directory: {OUT_DIR}")

    metadata_list = []

    for idx in range(N_SAMPLES):
        rng = np.random.default_rng(seed=idx + 42)
        sample_id = f"sample_{idx:04d}"
        h5_path = OUT_DIR / f"{sample_id}.h5"

        print(f"  [{idx+1}/{N_SAMPLES}] Building {sample_id} ...")

        mu = float(rng.uniform(2.5, 4.5))
        sigma_blur = float(rng.uniform(1.0, 2.5))

        x_true = _make_xray_phantom(rng)                             # (384,288)
        y_ideal = _fluoroscopy_forward(x_true, mu=mu, sigma_blur=sigma_blur)  # (384,288)

        assert x_true.shape == (IMG_H, IMG_W)
        assert y_ideal.shape == (IMG_H, IMG_W)

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("x_true", data=x_true, compression="gzip")
            hf.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = hf.create_group("H_params")
            hp.attrs["forward_model"] = "Beer-Lambert: y = exp(-mu * x_true) + Gaussian blur"
            hp.attrs["mu"] = mu
            hp.attrs["sigma_blur_px"] = sigma_blur
            hp.attrs["img_height"] = IMG_H
            hp.attrs["img_width"] = IMG_W

            meta = hf.create_group("metadata")
            meta.attrs["modality"] = MODALITY
            meta.attrs["sample_id"] = sample_id
            meta.attrs["x_shape"] = str([IMG_H, IMG_W])
            meta.attrs["y_shape"] = str([IMG_H, IMG_W])
            meta.attrs["seed"] = idx + 42
            meta.attrs["created"] = datetime.utcnow().isoformat()
            meta.attrs["source"] = "Synthetic chest/abdomen X-ray phantom"
            meta.attrs["reference"] = "Bernal et al. CMIG 2015 doi:10.1016/j.compmedimag.2015.02.007"

        metadata_list.append({
            "sample_id": sample_id,
            "h5_file": f"{sample_id}.h5",
            "x_shape": [IMG_H, IMG_W],
            "y_shape": [IMG_H, IMG_W],
            "mu": mu,
            "sigma_blur_px": sigma_blur,
            "seed": idx + 42,
        })

    meta_doc = {
        "modality": MODALITY,
        "n_samples": N_SAMPLES,
        "split": "standard",
        "created": datetime.utcnow().isoformat(),
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY}/standard/",
        "source": "Synthetic chest/abdomen X-ray phantom (bones, soft tissue, lungs)",
        "reference": "Bernal et al. CMIG 2015, doi:10.1016/j.compmedimag.2015.02.007",
        "samples": metadata_list,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta_doc, f, indent=2)

    spec_doc = {
        "modality": MODALITY,
        "x_description": "2-D X-ray attenuation image, shape (384,288), float32 [0,1]",
        "y_description": (
            "Fluoroscopic measurement via Beer-Lambert + Gaussian blur, "
            "shape (384,288), float32 [0,1]"
        ),
        "forward_model": "y = Gaussian_blur(exp(-mu * x_true)); mu in [2.5, 4.5], sigma in [1.0, 2.5] px",
        "x_shape": [IMG_H, IMG_W],
        "y_shape": [IMG_H, IMG_W],
        "dtype": "float32",
        "normalisation": "[0,1] for both x_true and y_ideal",
        "anatomical_features": ["chest wall", "ribs", "spine", "lungs", "soft-tissue organs",
                                 "bone inclusions"],
        "reconstruction_algorithms": [
            "Beer-Lambert inversion", "deconvolution", "total variation denoising",
            "U-Net fluoroscopy enhancer"
        ],
        "reference": "Bernal et al. CMIG 2015, doi:10.1016/j.compmedimag.2015.02.007",
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY}/standard/",
    }
    with open(OUT_DIR / "spec.json", "w") as f:
        json.dump(spec_doc, f, indent=2)

    print(f"[fluoroscopy] Done. {N_SAMPLES} samples written to {OUT_DIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
