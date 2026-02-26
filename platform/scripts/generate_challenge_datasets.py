#!/usr/bin/env python3
"""Generate HDF5 challenge datasets for the Blind Reconstruction Challenge.

Creates Public, Dev, and Hidden HDF5 files for each variant.
All three tiers use ALL scenes but with different mismatch realizations
(different true_spec values + different noise seeds).

Public HDF5 schema (what contestants download — includes ground truth):
    /sample_{nn}/y           — measurements (corrupted by mismatch + noise)
    /sample_{nn}/H_ideal     — ideal operator components
    /sample_{nn}/spec_ranges — JSON string with mismatch ranges
    /sample_{nn}/metadata    — JSON string (scene name, dimensions, noise model)
    /sample_{nn}/x_true      — ground truth signal
    /sample_{nn}/true_spec   — JSON string with exact mismatch params

Dev HDF5 schema (contestants download — no ground truth):
    /sample_{nn}/y           — measurements (corrupted by mismatch + noise)
    /sample_{nn}/H_ideal     — ideal operator components
    /sample_{nn}/spec_ranges — JSON string with mismatch ranges
    /sample_{nn}/metadata    — JSON string (scene name, dimensions, noise model)

Hidden HDF5 schema (server-side only — includes ground truth for eval):
    /sample_{nn}/...         — same as Public (full data for server-side evaluation)

Usage:
    python scripts/generate_challenge_datasets.py --variant sd_cassi
    python scripts/generate_challenge_datasets.py --variant all
    python scripts/generate_challenge_datasets.py --variant sd_cassi --output-dir /tmp/challenge
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

# Add parent to path so we can import challenge config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pwm_platform.services.benchmark_database._challenge_data import CHALLENGE_CONFIG

logger = logging.getLogger(__name__)

# ── Data loaders ──────────────────────────────────────────────────────────────


def _load_mat_scene(path: Path, key: str | None = None) -> np.ndarray:
    """Load a .mat file and return the signal array."""
    import scipy.io as sio

    data = sio.loadmat(str(path))
    if key is not None:
        return np.array(data[key], dtype=np.float64)
    # Auto-detect: pick the largest non-metadata array
    candidates = {
        k: v for k, v in data.items()
        if not k.startswith("_") and isinstance(v, np.ndarray)
    }
    if not candidates:
        raise ValueError(f"No arrays found in {path}")
    best = max(candidates, key=lambda k: candidates[k].size)
    return np.array(candidates[best], dtype=np.float64)


def _load_tif_image(path: Path) -> np.ndarray:
    """Load a .tif image and return as float64 in [0, 1]."""
    from PIL import Image

    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float64) / 255.0


# ── Mismatch application ─────────────────────────────────────────────────────


def _apply_cassi_mismatch(
    x: np.ndarray, mask: np.ndarray, true_spec: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Apply CASSI mismatch: subpixel mask shift, rotation, modified dispersion.

    Returns (y_mismatch, H_ideal_mask).
    """
    from scipy.ndimage import shift, rotate

    H, W, L = x.shape

    # Generate ideal coded aperture mask if not provided
    if mask is None:
        rng = np.random.default_rng(42)
        mask = (rng.random((H, W)) > 0.5).astype(np.float64)

    # Ideal measurement (no mismatch)
    # CASSI: y = sum_l mask * shift(x_l, l*dispersion)
    ideal_dispersion = 2.0  # nominal slope

    # Apply mismatch to mask: subpixel shift + rotation
    dx = true_spec["mask_dx"]
    dy = true_spec["mask_dy"]
    rot = true_spec["mask_rotation"]
    mismatch_mask = shift(mask, [dy, dx], order=1, mode="constant")
    if abs(rot) > 1e-6:
        mismatch_mask = rotate(mismatch_mask, rot, reshape=False, order=1, mode="constant")

    # Apply mismatch dispersion
    mismatch_slope = true_spec["dispersion_slope"]
    mismatch_axis = true_spec["dispersion_axis"]

    # Generate mismatched measurement
    y = np.zeros((H, W + (L - 1) * int(np.ceil(mismatch_slope))), dtype=np.float64)
    for l in range(L):
        disp = mismatch_slope * l
        disp_int = int(np.floor(disp))
        disp_frac = disp - disp_int
        coded = mismatch_mask * x[:, :, l]
        # Sub-pixel dispersion via linear interpolation
        y[:, disp_int:disp_int + W] += coded * (1 - disp_frac)
        if disp_frac > 0:
            y[:, disp_int + 1:disp_int + 1 + W] += coded * disp_frac

    return y, mask


def _apply_cacti_mismatch(
    x: np.ndarray, mask: np.ndarray, true_spec: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Apply CACTI mismatch: mask shift/rotation/blur, temporal clock offset, gain/offset drift.

    Returns (y_mismatch, H_ideal_mask).
    """
    from scipy.ndimage import shift, rotate, gaussian_filter

    H, W, T = x.shape

    # Generate ideal temporal mask if not provided
    if mask is None:
        rng = np.random.default_rng(42)
        mask = (rng.random((H, W, T)) > 0.5).astype(np.float64)

    # Apply mask mismatch
    mismatch_mask = mask.copy()
    dx = true_spec["mask_dx"]
    dy = true_spec["mask_dy"]
    rot = true_spec["mask_rotation"]
    blur = true_spec["mask_blur"]

    for t in range(T):
        frame = mismatch_mask[:, :, t]
        frame = shift(frame, [dy, dx], order=1, mode="constant")
        if abs(rot) > 1e-6:
            frame = rotate(frame, rot, reshape=False, order=1, mode="constant")
        if blur > 0:
            frame = gaussian_filter(frame, sigma=blur)
        mismatch_mask[:, :, t] = frame

    # Generate measurement: y = sum_t mask_t * x_t + gain/offset drift
    gain = true_spec["gain_drift"]
    offset = true_spec["offset_drift"]
    y = np.zeros((H, W), dtype=np.float64)
    for t in range(T):
        y += mismatch_mask[:, :, t] * x[:, :, t]
    y = gain * y + offset

    return y, mask


def _apply_spc_mismatch(
    x: np.ndarray, phi: np.ndarray, true_spec: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SPC mismatch: exponential gain decay + Gaussian noise.

    Returns (y_mismatch, H_ideal_matrix).
    """
    n = x.size
    x_flat = x.flatten()

    # Generate ideal sensing matrix if not provided
    if phi is None:
        rng = np.random.default_rng(42)
        m = n // 4  # 25% compression ratio
        phi = (rng.random((m, n)) > 0.5).astype(np.float64)

    m = phi.shape[0]

    # Ideal measurement
    y_ideal = phi @ x_flat

    # Apply gain decay: diag(exp(-alpha * i)) * y
    alpha = true_spec["gain_decay_alpha"]
    decay = np.exp(-alpha * np.arange(m))
    y = decay * y_ideal

    return y, phi


# ── Noise application ─────────────────────────────────────────────────────────


def _add_noise(y: np.ndarray, noise_model: str, noise_params: dict, rng: np.random.Generator) -> np.ndarray:
    """Add noise to measurements."""
    if noise_model == "poisson_gaussian":
        alpha = noise_params.get("poisson_alpha", 1.0)
        sigma = noise_params.get("gaussian_sigma", 0.01)
        # Poisson component (scaled)
        y_pos = np.maximum(y, 0)
        if alpha > 0 and y_pos.max() > 0:
            y_noisy = rng.poisson(np.maximum(y_pos / alpha, 0.001)).astype(np.float64) * alpha
        else:
            y_noisy = y.copy()
        # Gaussian component
        y_noisy += rng.normal(0, sigma, y.shape)
        return y_noisy
    elif noise_model == "gaussian":
        sigma = noise_params.get("sigma", 0.03)
        return y + rng.normal(0, sigma, y.shape)
    else:
        return y.copy()


# ── HDF5 writer ───────────────────────────────────────────────────────────────


def _write_sample(
    grp: h5py.Group,
    y: np.ndarray,
    H_ideal: np.ndarray,
    spec_ranges: list[dict],
    metadata: dict,
    x_true: np.ndarray | None = None,
    true_spec: dict | None = None,
):
    """Write a single sample to an HDF5 group."""
    grp.create_dataset("y", data=y, compression="gzip", compression_opts=4)
    grp.create_dataset("H_ideal", data=H_ideal, compression="gzip", compression_opts=4)
    grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
    grp.attrs["metadata"] = json.dumps(metadata)

    # Ground-truth fields (included in Public + Hidden tiers)
    if x_true is not None:
        grp.create_dataset("x_true", data=x_true, compression="gzip", compression_opts=4)
    if true_spec is not None:
        grp.attrs["true_spec"] = json.dumps(true_spec)


# ── Per-variant generators ────────────────────────────────────────────────────


def _find_data_root() -> Path:
    """Find the project datasets directory."""
    # Try relative to script location (platform/scripts/ -> ../../datasets/)
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "datasets",
        Path.cwd() / "datasets",
        Path.cwd().parent / "datasets",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "Cannot find datasets/ directory. "
        "Run from the project root or set --data-root."
    )


def _include_ground_truth(tier_name: str, visible_data: list[str]) -> bool:
    """Determine whether to include ground truth in the HDF5 file.

    Public tier: always includes x_true + true_spec (visible to contestants).
    Hidden tier: always includes x_true + true_spec (for server-side eval).
    Dev tier: never includes ground truth.
    """
    if tier_name == "dev":
        return False
    return True


def _generate_cassi(cfg: dict, output_dir: Path, data_root: Path | None = None):
    """Generate SD-CASSI challenge datasets (3 tiers)."""
    if data_root is None:
        data_root = _find_data_root()
    truth_dir = data_root / "TSA_simu_data" / "Truth"

    scenes = cfg["scenes"]

    for tier_name, tier_cfg in cfg["tiers"].items():
        tier_true_spec = tier_cfg["true_spec"]
        tier_seed = tier_cfg["seed"]
        visible_data = tier_cfg["visible_data"]

        rng = np.random.default_rng(tier_seed)
        mask = None  # Will be generated on first use

        out_path = output_dir / f"sd_cassi_challenge_{tier_name}.h5"
        logger.info("Generating %s -> %s", tier_name, out_path)

        with h5py.File(out_path, "w") as f:
            f.attrs["variant"] = "sd_cassi"
            f.attrs["tier"] = tier_name
            f.attrs["version"] = "1.0"

            for i, scene_id in enumerate(scenes):
                scene_path = truth_dir / f"scene{scene_id:02d}.mat"
                if not scene_path.exists():
                    logger.warning("Scene file not found: %s, skipping", scene_path)
                    continue

                x = _load_mat_scene(scene_path)
                # Normalize to [0, 1]
                if x.max() > 0:
                    x = x / x.max()

                # Generate mask once per tier
                if mask is None:
                    H, W = x.shape[:2]
                    mask = (rng.random((H, W)) > 0.5).astype(np.float64)

                y, H_ideal = _apply_cassi_mismatch(x, mask, tier_true_spec)
                y = _add_noise(y, cfg["noise_model"], cfg["noise_params"], rng)

                include_gt = _include_ground_truth(tier_name, visible_data)

                grp = f.create_group(f"sample_{i:02d}")
                _write_sample(
                    grp, y, H_ideal, cfg["spec_ranges"],
                    metadata={
                        "scene": f"scene{scene_id:02d}",
                        "shape": list(x.shape),
                        "noise_model": cfg["noise_model"],
                    },
                    x_true=x if include_gt else None,
                    true_spec=tier_true_spec if include_gt else None,
                )

        logger.info("Written %d samples to %s", len(scenes), out_path)


def _generate_cacti(cfg: dict, output_dir: Path, data_root: Path | None = None):
    """Generate CACTI challenge datasets (3 tiers)."""
    if data_root is None:
        data_root = _find_data_root()
    sim_dir = data_root / "CACTI" / "simulation"

    scenes = cfg["scenes"]

    for tier_name, tier_cfg in cfg["tiers"].items():
        tier_true_spec = tier_cfg["true_spec"]
        tier_seed = tier_cfg["seed"]
        visible_data = tier_cfg["visible_data"]

        rng = np.random.default_rng(tier_seed)
        mask = None

        out_path = output_dir / f"cacti_challenge_{tier_name}.h5"
        logger.info("Generating %s -> %s", tier_name, out_path)

        with h5py.File(out_path, "w") as f:
            f.attrs["variant"] = "cacti"
            f.attrs["tier"] = tier_name
            f.attrs["version"] = "1.0"

            for i, scene_name in enumerate(scenes):
                scene_path = sim_dir / f"{scene_name}.mat"
                if not scene_path.exists():
                    logger.warning("Scene file not found: %s, skipping", scene_path)
                    continue

                x = _load_mat_scene(scene_path)
                if x.max() > 0:
                    x = x / x.max()

                if mask is None:
                    H, W, T = x.shape
                    mask = (rng.random((H, W, T)) > 0.5).astype(np.float64)

                y, H_ideal = _apply_cacti_mismatch(x, mask, tier_true_spec)
                y = _add_noise(y, cfg["noise_model"], cfg["noise_params"], rng)

                include_gt = _include_ground_truth(tier_name, visible_data)

                grp = f.create_group(f"sample_{i:02d}")
                _write_sample(
                    grp, y, H_ideal, cfg["spec_ranges"],
                    metadata={
                        "scene": scene_name,
                        "shape": list(x.shape),
                        "noise_model": cfg["noise_model"],
                    },
                    x_true=x if include_gt else None,
                    true_spec=tier_true_spec if include_gt else None,
                )

        logger.info("Written %d samples to %s", len(scenes), out_path)


def _generate_spc(variant_key: str, cfg: dict, output_dir: Path, data_root: Path | None = None):
    """Generate SPC challenge datasets (block or kronecker, 3 tiers)."""
    if data_root is None:
        data_root = _find_data_root()
    set11_dir = data_root / "SPC" / "Set11"

    scenes = cfg["scenes"]

    # List .tif files sorted
    tif_files = sorted(set11_dir.glob("*.tif"))
    if not tif_files:
        tif_files = sorted(set11_dir.glob("*.png"))
    if not tif_files:
        logger.warning("No image files found in %s", set11_dir)
        return

    for tier_name, tier_cfg in cfg["tiers"].items():
        tier_true_spec = tier_cfg["true_spec"]
        tier_seed = tier_cfg["seed"]
        visible_data = tier_cfg["visible_data"]

        rng = np.random.default_rng(tier_seed)
        phi = None

        out_path = output_dir / f"{variant_key}_challenge_{tier_name}.h5"
        logger.info("Generating %s -> %s", tier_name, out_path)

        with h5py.File(out_path, "w") as f:
            f.attrs["variant"] = variant_key
            f.attrs["tier"] = tier_name
            f.attrs["version"] = "1.0"

            for i, scene_id in enumerate(scenes):
                idx = scene_id - 1  # 1-indexed to 0-indexed
                if idx >= len(tif_files):
                    logger.warning("Image index %d out of range, skipping", scene_id)
                    continue

                x = _load_tif_image(tif_files[idx])

                y, H_ideal = _apply_spc_mismatch(x, phi, tier_true_spec)
                y = _add_noise(y, cfg["noise_model"], cfg["noise_params"], rng)

                # Cache phi for reuse
                if phi is None:
                    phi = H_ideal

                include_gt = _include_ground_truth(tier_name, visible_data)

                grp = f.create_group(f"sample_{i:02d}")
                _write_sample(
                    grp, y, H_ideal, cfg["spec_ranges"],
                    metadata={
                        "scene": tif_files[idx].stem,
                        "shape": list(x.shape),
                        "noise_model": cfg["noise_model"],
                    },
                    x_true=x if include_gt else None,
                    true_spec=tier_true_spec if include_gt else None,
                )

        logger.info("Written %d samples to %s", len(scenes), out_path)


# ── Variant dispatch ──────────────────────────────────────────────────────────
_GENERATORS = {
    "sd_cassi": lambda cfg, out, dr: _generate_cassi(cfg, out, dr),
    "cacti": lambda cfg, out, dr: _generate_cacti(cfg, out, dr),
    "spc_block": lambda cfg, out, dr: _generate_spc("spc_block", cfg, out, dr),
    "spc_kronecker": lambda cfg, out, dr: _generate_spc("spc_kronecker", cfg, out, dr),
}


def generate_variant(variant_key: str, output_dir: Path, data_root: Path | None = None):
    """Generate challenge datasets for a single variant."""
    cfg = CHALLENGE_CONFIG.get(variant_key)
    if cfg is None:
        raise ValueError(f"No challenge config for variant: {variant_key}")

    gen = _GENERATORS.get(variant_key)
    if gen is None:
        raise ValueError(f"No generator for variant: {variant_key}")

    output_dir.mkdir(parents=True, exist_ok=True)
    gen(cfg, output_dir, data_root)


def main():
    parser = argparse.ArgumentParser(description="Generate challenge datasets")
    parser.add_argument(
        "--variant",
        required=True,
        help="Variant key (sd_cassi, cacti, spc_block, spc_kronecker) or 'all'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("challenge_datasets"),
        help="Output directory for HDF5 files (default: challenge_datasets/)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing source datasets (auto-detected if not set)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.variant == "all":
        variants = list(CHALLENGE_CONFIG.keys())
    else:
        variants = [args.variant]

    for v in variants:
        logger.info("=== Generating challenge datasets for %s ===", v)
        try:
            generate_variant(v, args.output_dir, args.data_root)
        except FileNotFoundError as e:
            logger.error("Skipping %s: %s", v, e)
        except Exception:
            logger.exception("Failed to generate %s", v)

    logger.info("Done.")


if __name__ == "__main__":
    main()
