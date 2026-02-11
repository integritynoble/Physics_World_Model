"""Generate InverseNet dataset for the Single-Pixel Camera (SPC) modality.

Sweep axes
----------
- compression_ratio (CR) : {0.10, 0.25, 0.50}
- photon_level           : {1e3, 1e4, 1e5}
- mismatch_family        : {"gain", "mask_error"}
- severity               : {"mild", "moderate", "severe"}

Each sample emits:
  x_gt          -- ground truth image  (H, W)
  y_clean       -- clean measurement   (M,)
  y             -- noisy + mismatched measurement
  mask          -- measurement matrix   (M, N)
  theta.json    -- true operator params
  delta_theta   -- applied mismatch
  y_cal         -- calibration captures (a few known-pattern measurements)

Output layout (per modality)::

    inversenet_spc/
        manifest.jsonl
        samples/
            spc_cr010_p1e3_gain_mild_s42/
                x_gt.npy
                y.npy
                y_clean.npy
                mask.npy
                theta.json
                delta_theta.json
                y_cal.npy
                runbundle_manifest.json

Usage::

    python -m experiments.inversenet.gen_spc --out_dir ./datasets/inversenet_spc
    python -m experiments.inversenet.gen_spc --smoke   # single sample
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.inversenet.manifest_schema import ManifestRecord, Modality, Severity
from experiments.inversenet.mismatch_sweep import (
    apply_mismatch,
    get_delta_theta,
)

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

IMAGE_SIZE = (64, 64)
COMPRESSION_RATIOS = [0.10, 0.25, 0.50]
PHOTON_LEVELS = [1e3, 1e4, 1e5]
MISMATCH_FAMILIES = ["gain", "mask_error"]
SEVERITIES = [Severity.mild, Severity.moderate, Severity.severe]
BASE_SEED = 42
N_CAL_PATTERNS = 5


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


# Real dataset path (Set11: 11 grayscale 256x256 .tif images)
_SET11_DIR = "/home/spiritai/ISTA-Net-PyTorch-master/data/Set11"
_SET11_IMAGES: Optional[List[np.ndarray]] = None


def _load_set11() -> List[np.ndarray]:
    """Load Set11 images as float32 [0,1] arrays, resized to IMAGE_SIZE."""
    global _SET11_IMAGES
    if _SET11_IMAGES is not None:
        return _SET11_IMAGES
    from PIL import Image
    imgs = []
    if os.path.isdir(_SET11_DIR):
        for f in sorted(os.listdir(_SET11_DIR)):
            if f.endswith((".tif", ".png", ".bmp")):
                img = Image.open(os.path.join(_SET11_DIR, f)).convert("L")
                img = img.resize(IMAGE_SIZE, Image.BICUBIC)
                arr = np.array(img, dtype=np.float32) / 255.0
                imgs.append(arr)
        logger.info("Loaded %d images from Set11 (%s)", len(imgs), _SET11_DIR)
    if not imgs:
        logger.warning("Set11 not found at %s; using synthetic fallback", _SET11_DIR)
    _SET11_IMAGES = imgs
    return imgs


def _make_ground_truth(rng: np.random.Generator) -> np.ndarray:
    """Load a real image from Set11, or fall back to synthetic."""
    real = _load_set11()
    if real:
        idx = int(rng.integers(0, len(real)))
        return real[idx].copy()
    # Synthetic fallback
    from scipy.ndimage import gaussian_filter
    raw = rng.random(IMAGE_SIZE).astype(np.float32)
    smooth = gaussian_filter(raw, sigma=5.0)
    smooth -= smooth.min()
    smooth /= smooth.max() + 1e-8
    return smooth.astype(np.float32)


def _build_measurement_matrix(
    cr: float, seed: int
) -> np.ndarray:
    """Build a Hadamard-like random +/- 1 measurement matrix."""
    rng = np.random.default_rng(seed)
    N = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    M = max(1, int(N * cr))
    A = (rng.random((M, N)) > 0.5).astype(np.float32) * 2 - 1
    A /= np.sqrt(N)
    return A


def _apply_photon_noise(
    y: np.ndarray, photon_level: float, rng: np.random.Generator
) -> np.ndarray:
    """Poisson + Gaussian noise (canonical implementation)."""
    from pwm_core.noise.apply import apply_photon_noise
    return apply_photon_noise(y, photon_level, rng)


def _generate_calibration_captures(
    A: np.ndarray, n_patterns: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate calibration measurements from known flat-field patterns."""
    N = A.shape[1]
    cal_patterns = []
    for i in range(n_patterns):
        # Flat fields with different intensity levels
        level = (i + 1) / n_patterns
        x_cal = np.full(N, level, dtype=np.float32)
        y_cal = A @ x_cal
        cal_patterns.append(y_cal)
    return np.array(cal_patterns, dtype=np.float32)


# ── Main generation ────────────────────────────────────────────────────

def generate_spc_sample(
    cr: float,
    photon_level: float,
    mismatch_family: str,
    severity: Severity,
    seed: int,
    out_dir: str,
) -> ManifestRecord:
    """Generate one SPC sample and write artifacts to disk."""
    rng = np.random.default_rng(seed)

    # Sample ID
    cr_tag = f"cr{int(cr * 100):03d}"
    photon_tag = f"p{photon_level:.0e}".replace("+", "")
    sid = f"spc_{cr_tag}_{photon_tag}_{mismatch_family}_{severity.value}_s{seed}"
    sample_dir = os.path.join(out_dir, "samples", sid)
    os.makedirs(sample_dir, exist_ok=True)

    # Ground truth
    x_gt = _make_ground_truth(rng)

    # Measurement matrix
    A = _build_measurement_matrix(cr, seed)

    # Clean measurement
    y_clean = A @ x_gt.flatten()

    # Photon noise
    y_noisy = _apply_photon_noise(y_clean, photon_level, rng)

    # Apply mismatch
    theta_true = {"gain": 1.0, "bias": 0.0, "cr": cr}
    mm = apply_mismatch(
        "spc",
        mismatch_family,
        severity,
        y=y_noisy,
        mask=A,
        rng=rng,
    )
    delta_theta = mm["delta_theta"]

    # Mismatched measurement / mask
    y_final = mm.get("y", y_noisy)
    A_final = mm.get("mask", A)

    # Calibration captures
    y_cal = _generate_calibration_captures(A, N_CAL_PATTERNS, rng)

    # Save artifacts
    np.save(os.path.join(sample_dir, "x_gt.npy"), x_gt)
    np.save(os.path.join(sample_dir, "y.npy"), y_final)
    np.save(os.path.join(sample_dir, "y_clean.npy"), y_clean)
    np.save(os.path.join(sample_dir, "mask.npy"), A_final)
    np.save(os.path.join(sample_dir, "y_cal.npy"), y_cal)

    with open(os.path.join(sample_dir, "theta.json"), "w") as f:
        json.dump(theta_true, f, indent=2)
    with open(os.path.join(sample_dir, "delta_theta.json"), "w") as f:
        json.dump(delta_theta, f, indent=2)

    # RunBundle manifest
    artifacts = {
        "x_gt": "x_gt.npy",
        "y": "y.npy",
        "x_hat": "x_gt.npy",  # placeholder (no recon yet)
    }
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

    # Build manifest record
    rel_prefix = os.path.join("samples", sid)
    record = ManifestRecord(
        sample_id=sid,
        modality=Modality.spc,
        seed=seed,
        photon_level=photon_level,
        compression_ratio=cr,
        mismatch_family=mismatch_family,
        severity=Severity(severity.value),
        theta=theta_true,
        delta_theta=delta_theta,
        paths={
            "x_gt": os.path.join(rel_prefix, "x_gt.npy"),
            "y": os.path.join(rel_prefix, "y.npy"),
            "y_clean": os.path.join(rel_prefix, "y_clean.npy"),
            "mask": os.path.join(rel_prefix, "mask.npy"),
            "y_cal": os.path.join(rel_prefix, "y_cal.npy"),
            "theta": os.path.join(rel_prefix, "theta.json"),
            "delta_theta": os.path.join(rel_prefix, "delta_theta.json"),
        },
        git_hash=_git_hash(),
        pwm_version="0.3.0",
    )
    return record


def generate_spc_dataset(
    out_dir: str,
    smoke: bool = False,
) -> List[ManifestRecord]:
    """Generate the full SPC sweep and write manifest.jsonl."""
    os.makedirs(out_dir, exist_ok=True)
    records: List[ManifestRecord] = []

    if smoke:
        combos = [(COMPRESSION_RATIOS[0], PHOTON_LEVELS[1],
                    MISMATCH_FAMILIES[0], SEVERITIES[0])]
    else:
        combos = list(product(
            COMPRESSION_RATIOS, PHOTON_LEVELS, MISMATCH_FAMILIES, SEVERITIES
        ))

    for idx, (cr, photon, fam, sev) in enumerate(combos):
        seed = BASE_SEED + idx
        logger.info(
            f"SPC [{idx + 1}/{len(combos)}] "
            f"CR={cr} photon={photon:.0e} {fam}/{sev.value}"
        )
        rec = generate_spc_sample(cr, photon, fam, sev, seed, out_dir)
        records.append(rec)

    # Write manifest
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")

    logger.info(f"SPC dataset: {len(records)} samples -> {manifest_path}")
    return records


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate InverseNet SPC dataset")
    parser.add_argument("--out_dir", default="datasets/inversenet_spc")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick validation run (1 sample)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_spc_dataset(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
