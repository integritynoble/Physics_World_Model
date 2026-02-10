"""Generate InverseNet dataset for CACTI (Coded Aperture Compressive Temporal Imaging).

Sweep axes
----------
- n_frames           : {4, 8, 16}
- photon_level       : {1e3, 1e4, 1e5}
- mismatch_family    : {"mask_shift", "temporal_jitter"}
- severity           : {"mild", "moderate", "severe"}

Each sample emits:
  x_gt          -- ground truth video cube (H, W, T)
  y             -- noisy + mismatched 2D snapshot (H, W)
  y_clean       -- clean measurement
  masks         -- temporal masks (H, W, T)
  masks_mm      -- mismatched masks (H, W, T)
  theta.json    -- true operator params
  delta_theta   -- applied mismatch
  y_cal         -- calibration captures

Output layout::

    inversenet_cacti/
        manifest.jsonl
        samples/
            cacti_f04_p1e3_mask_shift_mild_s42/
                ...

Usage::

    python -m experiments.inversenet.gen_cacti --out_dir ./datasets/inversenet_cacti
    python -m experiments.inversenet.gen_cacti --smoke
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
FRAME_COUNTS = [4, 8, 16]
PHOTON_LEVELS = [1e3, 1e4, 1e5]
MISMATCH_FAMILIES = ["mask_shift", "temporal_jitter"]
SEVERITIES = [Severity.mild, Severity.moderate, Severity.severe]
BASE_SEED = 1000
N_CAL_FRAMES = 3


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


# Real dataset: 6 grayscale CACTI benchmark videos (.mat with 'orig' key)
_CACTI_DIR = "/home/spiritai/PnP-SCI_python-master/dataset/cacti/grayscale_benchmark"
_CACTI_VIDEOS: Optional[List[np.ndarray]] = None


def _load_cacti_benchmark() -> List[np.ndarray]:
    """Load CACTI benchmark videos as float32 [0,1] arrays (H,W,T)."""
    global _CACTI_VIDEOS
    if _CACTI_VIDEOS is not None:
        return _CACTI_VIDEOS
    import scipy.io as sio
    vids = []
    if os.path.isdir(_CACTI_DIR):
        for f in sorted(os.listdir(_CACTI_DIR)):
            if f.endswith(".mat"):
                d = sio.loadmat(os.path.join(_CACTI_DIR, f))
                if "orig" in d:
                    orig = d["orig"].astype(np.float32)
                    if orig.max() > 1.0:
                        orig /= orig.max()
                    vids.append(orig)
        logger.info("Loaded %d CACTI videos from %s", len(vids), _CACTI_DIR)
    if not vids:
        logger.warning("CACTI benchmark not found at %s; using synthetic fallback", _CACTI_DIR)
    _CACTI_VIDEOS = vids
    return vids


def _make_video_gt(
    H: int, W: int, T: int, rng: np.random.Generator
) -> np.ndarray:
    """Load a real CACTI video, or fall back to synthetic."""
    from scipy.ndimage import zoom, gaussian_filter
    real = _load_cacti_benchmark()
    if real:
        idx = int(rng.integers(0, len(real)))
        vid = real[idx]  # (H_orig, W_orig, T_orig)
        # Resize spatial dims and select T frames
        T_avail = vid.shape[2]
        t_start = int(rng.integers(0, max(1, T_avail - T + 1))) if T_avail >= T else 0
        sel = vid[:, :, t_start:t_start + min(T, T_avail)]
        # Spatial resize
        if sel.shape[0] != H or sel.shape[1] != W:
            factors = (H / sel.shape[0], W / sel.shape[1], 1.0)
            sel = zoom(sel, factors, order=1)
        # Pad if fewer frames than T
        if sel.shape[2] < T:
            pad = np.repeat(sel[:, :, -1:], T - sel.shape[2], axis=2)
            sel = np.concatenate([sel, pad], axis=2)
        return np.clip(sel[:H, :W, :T], 0, 1).astype(np.float32)
    # Synthetic fallback
    video = np.zeros((H, W, T), dtype=np.float32)
    base = gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=5.0)
    base -= base.min()
    base /= base.max() + 1e-8
    for t in range(T):
        phase = 2.0 * np.pi * t / T
        shift_y = int(2 * np.sin(phase))
        shift_x = int(2 * np.cos(phase))
        frame = np.roll(np.roll(base, shift_y, axis=0), shift_x, axis=1)
        frame = frame * (0.8 + 0.2 * np.cos(phase))
        video[:, :, t] = frame
    return np.clip(video, 0, 1).astype(np.float32)


def _generate_masks(
    H: int, W: int, T: int, seed: int
) -> np.ndarray:
    """Generate binary temporal masks (H, W, T) via shifted base mask."""
    rng = np.random.default_rng(seed)
    base_mask = (rng.random((H, W)) > 0.5).astype(np.float32)
    masks = np.zeros((H, W, T), dtype=np.float32)
    for t in range(T):
        masks[:, :, t] = np.roll(base_mask, t, axis=0)
    return masks


def _cacti_forward(
    video: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    """CACTI forward: y = sum_t(mask_t * x_t)."""
    return np.sum(video * masks, axis=2).astype(np.float32)


def _apply_photon_noise(
    y: np.ndarray, photon_level: float, rng: np.random.Generator
) -> np.ndarray:
    """Poisson + Gaussian noise scaled to photon_level."""
    scale = photon_level / (np.abs(y).max() + 1e-10)
    y_scaled = np.maximum(y * scale, 0)
    y_noisy = rng.poisson(y_scaled).astype(np.float32)
    read_sigma = np.sqrt(photon_level) * 0.01
    y_noisy += rng.normal(0, read_sigma, size=y.shape).astype(np.float32)
    y_noisy /= scale
    return y_noisy


def _generate_calibration_captures(
    masks: np.ndarray, n_cal: int, rng: np.random.Generator
) -> np.ndarray:
    """Calibration: measure known flat-field patterns."""
    H, W, T = masks.shape
    cal = []
    for i in range(n_cal):
        level = (i + 1) / n_cal
        flat = np.full((H, W, T), level, dtype=np.float32)
        y_cal = _cacti_forward(flat, masks)
        cal.append(y_cal)
    return np.array(cal, dtype=np.float32)


# ── Main generation ────────────────────────────────────────────────────

def generate_cacti_sample(
    n_frames: int,
    photon_level: float,
    mismatch_family: str,
    severity: Severity,
    seed: int,
    out_dir: str,
) -> ManifestRecord:
    """Generate one CACTI sample."""
    rng = np.random.default_rng(seed)
    H, W = SPATIAL_SIZE

    sid = (
        f"cacti_f{n_frames:02d}_p{photon_level:.0e}_"
        f"{mismatch_family}_{severity.value}_s{seed}"
    ).replace("+", "")
    sample_dir = os.path.join(out_dir, "samples", sid)
    os.makedirs(sample_dir, exist_ok=True)

    # Ground truth video
    x_gt = _make_video_gt(H, W, n_frames, rng)

    # True masks
    masks_true = _generate_masks(H, W, n_frames, seed)

    # Clean measurement
    y_clean = _cacti_forward(x_gt, masks_true)

    # Photon noise
    y_noisy = _apply_photon_noise(y_clean, photon_level, rng)

    # Apply mismatch -> get modified masks
    mm = apply_mismatch(
        "cacti",
        mismatch_family,
        severity,
        masks=masks_true,
        rng=rng,
    )
    delta_theta = mm["delta_theta"]
    masks_mm = mm.get("masks", masks_true)

    # Re-measure with mismatched masks (simulates actual mismatch scenario)
    y_mm = _cacti_forward(x_gt, masks_mm)
    y_mm = _apply_photon_noise(y_mm, photon_level, rng)

    # Calibration
    y_cal = _generate_calibration_captures(masks_true, N_CAL_FRAMES, rng)

    theta_true = {
        "n_frames": n_frames,
        "shift_type": "vertical",
        "spatial_size": list(SPATIAL_SIZE),
    }

    # Save
    np.save(os.path.join(sample_dir, "x_gt.npy"), x_gt)
    np.save(os.path.join(sample_dir, "y.npy"), y_mm)
    np.save(os.path.join(sample_dir, "y_clean.npy"), y_clean)
    np.save(os.path.join(sample_dir, "masks.npy"), masks_true)
    np.save(os.path.join(sample_dir, "masks_mm.npy"), masks_mm)
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
        modality=Modality.cacti,
        seed=seed,
        photon_level=photon_level,
        n_frames=n_frames,
        mismatch_family=mismatch_family,
        severity=Severity(severity.value),
        theta=theta_true,
        delta_theta=delta_theta,
        paths={
            "x_gt": os.path.join(rel, "x_gt.npy"),
            "y": os.path.join(rel, "y.npy"),
            "y_clean": os.path.join(rel, "y_clean.npy"),
            "masks": os.path.join(rel, "masks.npy"),
            "masks_mm": os.path.join(rel, "masks_mm.npy"),
            "y_cal": os.path.join(rel, "y_cal.npy"),
            "theta": os.path.join(rel, "theta.json"),
            "delta_theta": os.path.join(rel, "delta_theta.json"),
        },
        git_hash=_git_hash(),
        pwm_version="0.3.0",
    )
    return record


def generate_cacti_dataset(
    out_dir: str,
    smoke: bool = False,
) -> List[ManifestRecord]:
    """Generate the full CACTI sweep."""
    os.makedirs(out_dir, exist_ok=True)
    records: List[ManifestRecord] = []

    if smoke:
        combos = [(FRAME_COUNTS[0], PHOTON_LEVELS[1],
                    MISMATCH_FAMILIES[0], SEVERITIES[0])]
    else:
        combos = list(product(
            FRAME_COUNTS, PHOTON_LEVELS, MISMATCH_FAMILIES, SEVERITIES
        ))

    for idx, (nf, photon, fam, sev) in enumerate(combos):
        seed = BASE_SEED + idx
        logger.info(
            f"CACTI [{idx + 1}/{len(combos)}] "
            f"frames={nf} photon={photon:.0e} {fam}/{sev.value}"
        )
        rec = generate_cacti_sample(nf, photon, fam, sev, seed, out_dir)
        records.append(rec)

    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")

    logger.info(f"CACTI dataset: {len(records)} samples -> {manifest_path}")
    return records


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate InverseNet CACTI dataset")
    parser.add_argument("--out_dir", default="datasets/inversenet_cacti")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_cacti_dataset(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
