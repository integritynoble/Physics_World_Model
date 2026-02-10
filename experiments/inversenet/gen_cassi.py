"""Generate InverseNet dataset for CASSI (Coded Aperture Snapshot Spectral Imaging).

Sweep axes
----------
- n_bands            : {8, 16, 28}
- photon_level       : {1e3, 1e4, 1e5}
- mismatch_family    : {"disp_step", "mask_shift", "PSF_blur"}
- severity           : {"mild", "moderate", "severe"}

Each sample emits:
  x_gt          -- ground truth hyperspectral cube (H, W, L)
  y             -- noisy + mismatched 2D snapshot  (H, W)
  y_clean       -- clean measurement
  mask          -- coded aperture (H, W)
  theta.json    -- true operator params (dispersion polynomial, etc.)
  delta_theta   -- applied mismatch
  y_cal         -- calibration captures

Output layout::

    inversenet_cassi/
        manifest.jsonl
        samples/
            cassi_b08_p1e3_disp_step_mild_s42/
                ...

Usage::

    python -m experiments.inversenet.gen_cassi --out_dir ./datasets/inversenet_cassi
    python -m experiments.inversenet.gen_cassi --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from itertools import product
from typing import Any, Dict, List

import numpy as np

from experiments.inversenet.manifest_schema import ManifestRecord, Modality, Severity
from experiments.inversenet.mismatch_sweep import (
    apply_mismatch,
    get_delta_theta,
)

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

SPATIAL_SIZE = (64, 64)
BAND_COUNTS = [8, 16, 28]
PHOTON_LEVELS = [1e3, 1e4, 1e5]
MISMATCH_FAMILIES = ["disp_step", "mask_shift", "PSF_blur"]
SEVERITIES = [Severity.mild, Severity.moderate, Severity.severe]
BASE_SEED = 2000
N_CAL_PATTERNS = 4


# ── Helpers ─────────────────────────────────────────────────────────────

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256(arr: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(arr.tobytes()).hexdigest()


# Real dataset: 10 HSI scenes from TSA simulation data (256x256x28 .mat with 'img' key)
_TSA_DIR = "/home/spiritai/MST-main/datasets/TSA_simu_data/Truth"
_TSA_SCENES: Optional[List[np.ndarray]] = None


def _load_tsa_scenes() -> List[np.ndarray]:
    """Load TSA Truth HSI scenes as float32 [0,1] arrays (H,W,L)."""
    global _TSA_SCENES
    if _TSA_SCENES is not None:
        return _TSA_SCENES
    import scipy.io as sio
    scenes = []
    if os.path.isdir(_TSA_DIR):
        for f in sorted(os.listdir(_TSA_DIR)):
            if f.endswith(".mat"):
                d = sio.loadmat(os.path.join(_TSA_DIR, f))
                if "img" in d:
                    img = d["img"].astype(np.float32)
                    if img.max() > 1.0:
                        img /= img.max()
                    scenes.append(img)
        logger.info("Loaded %d HSI scenes from TSA (%s)", len(scenes), _TSA_DIR)
    if not scenes:
        logger.warning("TSA data not found at %s; using synthetic fallback", _TSA_DIR)
    _TSA_SCENES = scenes
    return scenes


def _make_hsi_gt(
    H: int, W: int, L: int, rng: np.random.Generator
) -> np.ndarray:
    """Load a real HSI scene from TSA, or fall back to synthetic."""
    from scipy.ndimage import zoom, gaussian_filter
    real = _load_tsa_scenes()
    if real:
        idx = int(rng.integers(0, len(real)))
        scene = real[idx]  # (H_orig, W_orig, L_orig)
        # Select L bands (uniformly spaced from available)
        L_avail = scene.shape[2]
        if L_avail >= L:
            band_idx = np.linspace(0, L_avail - 1, L, dtype=int)
            sel = scene[:, :, band_idx]
        else:
            # Pad with last band repeated
            sel = np.concatenate(
                [scene, np.repeat(scene[:, :, -1:], L - L_avail, axis=2)],
                axis=2,
            )
        # Spatial resize
        if sel.shape[0] != H or sel.shape[1] != W:
            factors = (H / sel.shape[0], W / sel.shape[1], 1.0)
            sel = zoom(sel, factors, order=1)
        return np.clip(sel[:H, :W, :L], 0, 1).astype(np.float32)
    # Synthetic fallback
    cube = np.zeros((H, W, L), dtype=np.float32)
    n_basis = min(3, L)
    bases = []
    for _ in range(n_basis):
        b = gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=6.0)
        b -= b.min()
        b /= b.max() + 1e-8
        bases.append(b)
    wavelengths = np.linspace(0, 1, L)
    for l_idx in range(L):
        w = wavelengths[l_idx]
        cube[:, :, l_idx] = (
            bases[0 % n_basis] * np.exp(-10 * (w - 0.3) ** 2)
            + bases[1 % n_basis] * np.exp(-10 * (w - 0.6) ** 2)
            + bases[2 % n_basis] * 0.2
        )
    cube -= cube.min()
    cube /= cube.max() + 1e-8
    return cube.astype(np.float32)


def _generate_coded_aperture(
    H: int, W: int, seed: int
) -> np.ndarray:
    """Generate binary coded aperture mask (H, W)."""
    rng = np.random.default_rng(seed)
    return (rng.random((H, W)) > 0.5).astype(np.float32)


def _cassi_forward(
    cube: np.ndarray,
    mask: np.ndarray,
    theta: Dict[str, Any],
) -> np.ndarray:
    """CASSI forward model: mask + dispersion + sum.

    Uses the same dispersion_shift function as the CASSI operator.
    """
    from pwm_core.physics.spectral.dispersion_models import dispersion_shift

    H, W, L = cube.shape
    y = np.zeros((H, W), dtype=np.float32)
    for l_idx in range(L):
        dx, dy = dispersion_shift(theta, band=l_idx)
        band = cube[:, :, l_idx]
        band_shifted = np.roll(
            np.roll(band, int(round(dy)), axis=0),
            int(round(dx)), axis=1,
        )
        y += band_shifted * mask
    return y


def _apply_photon_noise(
    y: np.ndarray, photon_level: float, rng: np.random.Generator
) -> np.ndarray:
    """Poisson + Gaussian noise."""
    scale = photon_level / (np.abs(y).max() + 1e-10)
    y_scaled = np.maximum(y * scale, 0)
    y_noisy = rng.poisson(y_scaled).astype(np.float32)
    read_sigma = np.sqrt(photon_level) * 0.01
    y_noisy += rng.normal(0, read_sigma, size=y.shape).astype(np.float32)
    y_noisy /= scale
    return y_noisy


def _generate_calibration_captures(
    mask: np.ndarray, theta: Dict[str, Any], L: int,
    n_cal: int, rng: np.random.Generator
) -> np.ndarray:
    """Calibration: narrowband flat-field patterns at different bands."""
    H, W = mask.shape
    cal = []
    selected_bands = np.linspace(0, L - 1, n_cal, dtype=int)
    for b in selected_bands:
        cube_cal = np.zeros((H, W, L), dtype=np.float32)
        cube_cal[:, :, b] = 1.0
        y_cal = _cassi_forward(cube_cal, mask, theta)
        cal.append(y_cal)
    return np.array(cal, dtype=np.float32)


# ── Default CASSI theta ────────────────────────────────────────────────

def _default_cassi_theta(L: int) -> Dict[str, Any]:
    """Nominal theta for CASSI: linear dispersion, no offset."""
    return {
        "dispersion_model": "poly",
        "disp_poly_x": [0.0, 1.0, 0.0],  # dx = 0 + 1*l + 0*l^2
        "disp_poly_y": [0.0, 0.0, 0.0],
        "L": L,
    }


# ── Main generation ────────────────────────────────────────────────────

def generate_cassi_sample(
    n_bands: int,
    photon_level: float,
    mismatch_family: str,
    severity: Severity,
    seed: int,
    out_dir: str,
) -> ManifestRecord:
    """Generate one CASSI sample."""
    rng = np.random.default_rng(seed)
    H, W = SPATIAL_SIZE

    sid = (
        f"cassi_b{n_bands:02d}_p{photon_level:.0e}_"
        f"{mismatch_family}_{severity.value}_s{seed}"
    ).replace("+", "")
    sample_dir = os.path.join(out_dir, "samples", sid)
    os.makedirs(sample_dir, exist_ok=True)

    # Ground truth HSI
    x_gt = _make_hsi_gt(H, W, n_bands, rng)

    # Coded aperture mask
    mask_true = _generate_coded_aperture(H, W, seed)

    # Nominal theta
    theta_true = _default_cassi_theta(n_bands)

    # Clean measurement
    y_clean = _cassi_forward(x_gt, mask_true, theta_true)

    # Photon noise
    y_noisy = _apply_photon_noise(y_clean, photon_level, rng)

    # Apply mismatch
    mm = apply_mismatch(
        "cassi",
        mismatch_family,
        severity,
        y=y_noisy,
        mask=mask_true,
        theta=theta_true,
        rng=rng,
    )
    delta_theta = mm["delta_theta"]

    # Determine the mismatched artifacts
    theta_mm = mm.get("theta", theta_true)
    mask_mm = mm.get("mask", mask_true)
    y_mm = mm.get("y", y_noisy)

    # If theta or mask changed, re-simulate the measurement
    if "theta" in mm or "mask" in mm:
        y_mm = _cassi_forward(x_gt, mask_mm, theta_mm)
        y_mm = _apply_photon_noise(y_mm, photon_level, rng)

    # Calibration captures (using true mask + theta)
    y_cal = _generate_calibration_captures(mask_true, theta_true, n_bands,
                                           N_CAL_PATTERNS, rng)

    # Save
    np.save(os.path.join(sample_dir, "x_gt.npy"), x_gt)
    np.save(os.path.join(sample_dir, "y.npy"), y_mm)
    np.save(os.path.join(sample_dir, "y_clean.npy"), y_clean)
    np.save(os.path.join(sample_dir, "mask.npy"), mask_true)
    np.save(os.path.join(sample_dir, "mask_mm.npy"), mask_mm)
    np.save(os.path.join(sample_dir, "y_cal.npy"), y_cal)

    with open(os.path.join(sample_dir, "theta.json"), "w") as f:
        json.dump(theta_true, f, indent=2)
    with open(os.path.join(sample_dir, "delta_theta.json"), "w") as f:
        json.dump(delta_theta, f, indent=2)

    # RunBundle
    artifacts = {"x_gt": "x_gt.npy", "y": "y.npy", "x_hat": "x_gt.npy"}
    hashes = {k: _sha256(np.load(os.path.join(sample_dir, v))) for k, v in artifacts.items()}
    bundle = {
        "version": "0.3.0",
        "spec_id": sid,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "git_hash": _git_hash(),
            "seeds": [seed],
            "platform": platform.platform(),
            "pwm_version": "0.3.0",
        },
        "metrics": {"psnr_db": 0.0, "ssim": 0.0, "runtime_s": 0.0},
        "artifacts": artifacts,
        "hashes": hashes,
    }
    with open(os.path.join(sample_dir, "runbundle_manifest.json"), "w") as f:
        json.dump(bundle, f, indent=2)

    rel = os.path.join("samples", sid)
    record = ManifestRecord(
        sample_id=sid,
        modality=Modality.cassi,
        seed=seed,
        photon_level=photon_level,
        n_bands=n_bands,
        mismatch_family=mismatch_family,
        severity=Severity(severity.value),
        theta=theta_true,
        delta_theta=delta_theta,
        paths={
            "x_gt": os.path.join(rel, "x_gt.npy"),
            "y": os.path.join(rel, "y.npy"),
            "y_clean": os.path.join(rel, "y_clean.npy"),
            "mask": os.path.join(rel, "mask.npy"),
            "mask_mm": os.path.join(rel, "mask_mm.npy"),
            "y_cal": os.path.join(rel, "y_cal.npy"),
            "theta": os.path.join(rel, "theta.json"),
            "delta_theta": os.path.join(rel, "delta_theta.json"),
        },
        git_hash=_git_hash(),
        pwm_version="0.3.0",
    )
    return record


def generate_cassi_dataset(
    out_dir: str,
    smoke: bool = False,
) -> List[ManifestRecord]:
    """Generate the full CASSI sweep."""
    os.makedirs(out_dir, exist_ok=True)
    records: List[ManifestRecord] = []

    if smoke:
        combos = [(BAND_COUNTS[0], PHOTON_LEVELS[1],
                    MISMATCH_FAMILIES[0], SEVERITIES[0])]
    else:
        combos = list(product(
            BAND_COUNTS, PHOTON_LEVELS, MISMATCH_FAMILIES, SEVERITIES
        ))

    for idx, (nb, photon, fam, sev) in enumerate(combos):
        seed = BASE_SEED + idx
        logger.info(
            f"CASSI [{idx + 1}/{len(combos)}] "
            f"bands={nb} photon={photon:.0e} {fam}/{sev.value}"
        )
        rec = generate_cassi_sample(nb, photon, fam, sev, seed, out_dir)
        records.append(rec)

    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")

    logger.info(f"CASSI dataset: {len(records)} samples -> {manifest_path}")
    return records


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate InverseNet CASSI dataset")
    parser.add_argument("--out_dir", default="datasets/inversenet_cassi")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_cassi_dataset(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
