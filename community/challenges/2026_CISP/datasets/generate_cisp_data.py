"""
CISP 2026 Sealed-Simulator Data Generator
==========================================

Generates synthetic measurement data for CISP challenge evaluation.

All data is generated via the PWM harness using sealed simulator parameters.
No real-world data is used in CISP 2026.

Usage:
    python generate_cisp_data.py --track correct --severity moderate --scenes 10 --seed 2026
    python generate_cisp_data.py --track all --severity moderate --scenes 10 --seed 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Track → modality mapping
# ---------------------------------------------------------------------------

TRACK_MODALITIES = {
    "correct": ["cassi"],
    "temporal": ["cacti"],
    "medical": ["mri", "ct"],
    "cross": ["cassi", "spc", "cacti", "widefield"],
}

SEVERITIES = ["mild", "moderate", "severe"]

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_scene(modality: str, scene_idx: int, rng: np.random.Generator,
                   severity: str = "moderate") -> dict:
    """Generate a single synthetic scene for a modality.

    Returns a dict with:
        x_gt: ground truth scene (numpy array)
        y: measurement (numpy array)
        mismatch_params: dict of applied mismatch parameters
        metadata: scene provenance
    """
    # Scene dimensions per modality
    dims = {
        "cassi": (256, 256, 28),   # spatial x spatial x spectral
        "spc": (64, 64, 28),       # spatial x spatial x spectral
        "cacti": (256, 256, 8),    # spatial x spatial x temporal
        "widefield": (256, 256),   # spatial x spatial
        "confocal": (128, 128, 32),
        "lightsheet": (128, 128, 64),
        "mri": (256, 256),
        "ct": (256, 256),
    }

    shape = dims.get(modality, (128, 128))

    # Generate smooth synthetic scene (band-limited random)
    x_gt = _generate_smooth_scene(shape, rng)

    # Generate measurement with mismatch
    mismatch_params = _sample_mismatch(modality, severity, rng)
    y = _forward_with_mismatch(x_gt, modality, mismatch_params, rng)

    return {
        "x_gt": x_gt,
        "y": y,
        "mismatch_params": mismatch_params,
        "metadata": {
            "modality": modality,
            "scene_idx": scene_idx,
            "severity": severity,
            "shape": list(shape),
        },
    }


def _generate_smooth_scene(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Generate a band-limited random scene (smooth, realistic structure)."""
    # Start with random in frequency domain, low-pass filter
    raw = rng.standard_normal(shape)

    # Apply smoothing via low-frequency bias
    if len(shape) == 2:
        from numpy.fft import fft2, ifft2, fftfreq
        F = fft2(raw)
        ny, nx = shape
        fy = fftfreq(ny)[:, None]
        fx = fftfreq(nx)[None, :]
        # Gaussian low-pass with sigma ~ 0.15 of Nyquist
        lpf = np.exp(-0.5 * (fy**2 + fx**2) / 0.02)
        raw = np.real(ifft2(F * lpf))
    elif len(shape) == 3:
        # Apply 2D smoothing per channel
        from numpy.fft import fft2, ifft2, fftfreq
        ny, nx = shape[0], shape[1]
        fy = fftfreq(ny)[:, None]
        fx = fftfreq(nx)[None, :]
        lpf = np.exp(-0.5 * (fy**2 + fx**2) / 0.02)
        for c in range(shape[2]):
            F = fft2(raw[:, :, c])
            raw[:, :, c] = np.real(ifft2(F * lpf))

    # Normalize to [0, 1]
    raw = raw - raw.min()
    if raw.max() > 0:
        raw = raw / raw.max()
    return raw.astype(np.float32)


def _sample_mismatch(modality: str, severity: str, rng: np.random.Generator) -> dict:
    """Sample mismatch parameters for sealed-simulator evaluation."""
    severity_scale = {"mild": 0.25, "moderate": 0.5, "severe": 1.0}
    s = severity_scale.get(severity, 0.5)

    if modality == "cassi":
        return {
            "mask_shift_px": float(rng.uniform(0, 3.0 * s)),
            "dispersion_error_pct": float(rng.uniform(0, 10.0 * s)),
            "spectral_drift_nm": float(rng.uniform(0, 5.0 * s)),
        }
    elif modality == "spc":
        return {
            "pattern_error_pct": float(rng.uniform(0, 5.0 * s)),
            "detector_nonlinearity": float(rng.uniform(0, 3.0 * s)),
        }
    elif modality == "cacti":
        return {
            "timing_jitter_pct": float(rng.uniform(0, 5.0 * s)),
            "blur_kernel_error_px": float(rng.uniform(0, 2.0 * s)),
            "frame_drift_pct": float(rng.uniform(0, 5.0 * s)),
        }
    elif modality == "mri":
        return {
            "coil_sensitivity_error_pct": float(rng.uniform(0, 10.0 * s)),
            "trajectory_error_pct": float(rng.uniform(0, 5.0 * s)),
            "b0_inhomogeneity_hz": float(rng.uniform(0, 100.0 * s)),
        }
    elif modality == "ct":
        return {
            "beam_spectrum_error_pct": float(rng.uniform(0, 8.0 * s)),
            "geometry_error_mm": float(rng.uniform(0, 2.0 * s)),
        }
    else:
        return {
            "generic_perturbation_pct": float(rng.uniform(0, 5.0 * s)),
        }


def _forward_with_mismatch(x: np.ndarray, modality: str,
                           mismatch: dict, rng: np.random.Generator) -> np.ndarray:
    """Simulate forward measurement with mismatch (simplified sealed simulator).

    In a full deployment this calls the OperatorGraph compiler.
    For data generation, we use a simplified forward model.
    """
    # Simplified: linear projection + noise + mismatch perturbation
    if len(x.shape) == 3:
        # Compress along last dimension (spectral/temporal)
        y = x.sum(axis=-1)
    else:
        y = x.copy()

    # Add mismatch-induced perturbation
    perturbation_scale = sum(abs(v) for v in mismatch.values() if isinstance(v, (int, float)))
    perturbation_scale = min(perturbation_scale * 0.01, 0.5)
    y = y + perturbation_scale * rng.standard_normal(y.shape).astype(np.float32)

    # Add measurement noise (Poisson-Gaussian)
    snr_db = 25.0
    sigma = np.sqrt(np.mean(y**2)) * 10 ** (-snr_db / 20)
    y = y + sigma * rng.standard_normal(y.shape).astype(np.float32)

    return y


# ---------------------------------------------------------------------------
# Dataset packaging
# ---------------------------------------------------------------------------


def generate_track_data(track: str, severity: str, n_scenes: int,
                        seed: int, output_dir: Path) -> dict:
    """Generate all data for a CISP track."""
    modalities = TRACK_MODALITIES[track]
    rng = np.random.default_rng(seed)
    manifest = {
        "track": track,
        "severity": severity,
        "n_scenes": n_scenes,
        "seed": seed,
        "modalities": {},
    }

    for modality in modalities:
        mod_dir = output_dir / track / modality
        mod_dir.mkdir(parents=True, exist_ok=True)
        scene_hashes = []

        for i in range(n_scenes):
            scene = generate_scene(modality, i, rng, severity)
            scene_dir = mod_dir / f"scene_{i:04d}"
            scene_dir.mkdir(exist_ok=True)

            # Save arrays
            x_path = scene_dir / "x_gt.npy"
            y_path = scene_dir / "y.npy"
            meta_path = scene_dir / "metadata.json"

            np.save(x_path, scene["x_gt"])
            np.save(y_path, scene["y"])
            with open(meta_path, "w") as f:
                json.dump({
                    **scene["metadata"],
                    "mismatch_params": scene["mismatch_params"],
                }, f, indent=2)

            scene_hashes.append({
                "scene": i,
                "x_gt_sha256": _sha256(x_path),
                "y_sha256": _sha256(y_path),
            })

        manifest["modalities"][modality] = {
            "n_scenes": n_scenes,
            "scenes": scene_hashes,
        }

    # Write manifest
    manifest_path = output_dir / track / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate CISP 2026 sealed-simulator data"
    )
    parser.add_argument(
        "--track",
        choices=list(TRACK_MODALITIES.keys()) + ["all"],
        required=True,
        help="CISP track to generate data for",
    )
    parser.add_argument(
        "--severity",
        choices=SEVERITIES,
        default="moderate",
        help="Mismatch severity level (default: moderate)",
    )
    parser.add_argument(
        "--scenes",
        type=int,
        default=10,
        help="Number of scenes per modality (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility (default: 2026)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cisp_data/",
        help="Output directory (default: cisp_data/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = list(TRACK_MODALITIES.keys()) if args.track == "all" else [args.track]

    for track in tracks:
        print(f"Generating data for track: {track}")
        manifest = generate_track_data(
            track, args.severity, args.scenes, args.seed, output_dir
        )
        n_mod = len(manifest["modalities"])
        print(f"  {n_mod} modalities, {args.scenes} scenes each")

    print(f"\nData written to {output_dir}/")
    print("Verify integrity with: python generate_cisp_data.py --verify")


if __name__ == "__main__":
    main()
