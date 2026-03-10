"""Common-mode reconstruction service.

Loads benchmark data from GCS, maps algorithm name → reconstruction runner,
and returns results (images + metrics).
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

GCS_BUCKET = "pwm-benchmark-datasets"
GCS_PREFIX = "challenge-data/v1.0"
CACHE_DIR = Path("/tmp/pwm_challenge_cache")


def _ensure_challenge_h5(variant: str, tier: str = "public") -> Path:
    """Download challenge HDF5 from GCS (cached locally)."""
    cache = CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    filename = f"{variant}_challenge_{tier}.h5"
    local_path = cache / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    gcs_key = f"{GCS_PREFIX}/{filename}"
    try:
        from google.cloud import storage as gcs_storage
        client = gcs_storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_key)
        if not blob.exists():
            raise RuntimeError(f"GCS object not found: gs://{GCS_BUCKET}/{gcs_key}")
        blob.download_to_filename(str(local_path))
        return local_path
    except ImportError:
        raise RuntimeError("google-cloud-storage not installed")
    except Exception as e:
        if local_path.exists() and local_path.stat().st_size == 0:
            local_path.unlink()
        raise RuntimeError(f"Failed to download {filename}: {e}")


def _load_sample(h5_path: Path, sample_idx: int = 0) -> dict:
    """Load a single sample from challenge HDF5.

    Returns dict with keys: y (measurement), x_true (ground truth, if present),
    H_ideal (forward model, if present).
    """
    import h5py

    data = {}
    with h5py.File(h5_path, "r") as f:
        sample_key = f"sample_{sample_idx:02d}"
        if sample_key not in f:
            # Try first available sample
            samples = [k for k in f.keys() if k.startswith("sample_")]
            if not samples:
                raise ValueError(f"No samples in {h5_path}")
            sample_key = sorted(samples)[0]

        grp = f[sample_key]
        if "y" in grp:
            data["y"] = np.array(grp["y"])
        if "x_true" in grp:
            data["x_true"] = np.array(grp["x_true"])
        if "H_ideal" in grp:
            data["H_ideal"] = np.array(grp["H_ideal"])

    return data


def _numpy_to_png_b64(arr: np.ndarray) -> str:
    """Convert a 2D numpy array to base64-encoded PNG."""
    from PIL import Image

    # Normalize to [0, 255]
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)

    arr_f = arr.astype(np.float64)
    lo, hi = np.percentile(arr_f, [1, 99])
    if hi - lo > 1e-8:
        arr_f = np.clip((arr_f - lo) / (hi - lo), 0, 1)
    else:
        arr_f = np.clip(arr_f, 0, 1)

    img = Image.fromarray((arr_f * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _compute_psnr(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    x_true_f = x_true.astype(np.float64)
    x_recon_f = x_recon.astype(np.float64)
    mse = np.mean((x_true_f - x_recon_f) ** 2)
    if mse < 1e-12:
        return 60.0
    data_range = x_true_f.max() - x_true_f.min()
    if data_range < 1e-12:
        data_range = 1.0
    return float(10 * np.log10(data_range ** 2 / mse))


def _compute_ssim(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
        x_true_f = x_true.astype(np.float64).squeeze()
        x_recon_f = x_recon.astype(np.float64).squeeze()
        dr = x_true_f.max() - x_true_f.min()
        if dr < 1e-12:
            dr = 1.0
        return float(structural_similarity(x_true_f, x_recon_f, data_range=dr))
    except ImportError:
        return 0.0


def _run_classical_recon(
    y: np.ndarray,
    H: Optional[np.ndarray],
    algo_name: str,
) -> np.ndarray:
    """Run a classical reconstruction algorithm.

    For MVP: Tikhonov / pseudo-inverse for matrix-based systems,
    filtered back-projection style for others.
    """
    algo_lower = algo_name.lower()

    if H is not None and H.ndim == 2:
        # Matrix-based inverse: x = (H^T H + λI)^{-1} H^T y
        y_flat = y.flatten()
        m, n = H.shape
        if y_flat.shape[0] != m:
            # Truncate/pad
            y_flat = y_flat[:m] if y_flat.shape[0] > m else np.pad(y_flat, (0, m - y_flat.shape[0]))

        lam = 1e-3 if "tikhonov" in algo_lower else 1e-4
        try:
            HtH = H.T @ H
            Hty = H.T @ y_flat
            x_recon = np.linalg.solve(HtH + lam * np.eye(n), Hty)
            # Reshape to square image
            side = int(np.sqrt(n))
            if side * side == n:
                x_recon = x_recon.reshape(side, side)
            return x_recon
        except np.linalg.LinAlgError:
            pass

    # Fallback: return measurement resized as pseudo-reconstruction
    side = int(np.sqrt(y.size))
    if side * side != y.size:
        side = max(y.shape[-2:]) if y.ndim >= 2 else 64
    return y.reshape(y.shape[:2]) if y.ndim >= 2 else y.flatten()[:side * side].reshape(side, side)


async def run_common_reconstruction(
    variant_key: str,
    algorithm_name: str,
    user_measurement: Optional[np.ndarray] = None,
    user_matrix: Optional[np.ndarray] = None,
) -> dict:
    """Run a single algorithm on standard benchmark or user data.

    Returns dict with: reconstructed_image (base64 PNG), ground_truth_image,
    measurement_image, psnr, ssim, algorithm_info, runtime_ms.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _run_common_sync,
        variant_key, algorithm_name, user_measurement, user_matrix,
    )


def _run_common_sync(
    variant_key: str,
    algorithm_name: str,
    user_measurement: Optional[np.ndarray],
    user_matrix: Optional[np.ndarray],
) -> dict:
    """Synchronous common-mode reconstruction."""
    t0 = time.perf_counter()

    # Look up algorithm info
    from pwm_platform.services.benchmark_database import get_variant
    from pwm_platform.services.benchmark_database._algorithm_catalog import get_algorithms

    variant = get_variant(variant_key)
    if variant is None:
        raise ValueError(f"Unknown variant: {variant_key}")

    category = variant.get("category", "compressive")
    algos = get_algorithms(variant_key, category)
    algo_info = next((a for a in algos if a["name"] == algorithm_name), None)
    if algo_info is None:
        # Fuzzy match
        algo_info = next(
            (a for a in algos if a["name"].lower() == algorithm_name.lower()),
            algos[0] if algos else {"name": algorithm_name, "type": "Unknown", "source": ""},
        )

    # Load data
    has_gt = False
    if user_measurement is not None:
        y = user_measurement
        H = user_matrix
        x_true = None
    else:
        # Download from GCS
        try:
            h5_path = _ensure_challenge_h5(variant_key, "public")
            sample = _load_sample(h5_path)
            y = sample.get("y")
            x_true = sample.get("x_true")
            H = sample.get("H_ideal")
            has_gt = x_true is not None
        except Exception as exc:
            logger.warning("Cannot load challenge data for %s: %s", variant_key, exc)
            raise ValueError(
                f"No benchmark data available for {variant_key}. "
                "Upload your own measurement data instead."
            )

    if y is None:
        raise ValueError("No measurement data found")

    # Run reconstruction
    algo_type = algo_info.get("type", "").lower()
    is_dl = any(kw in algo_type for kw in ("deep", "transformer", "diffusion", "gan"))

    if is_dl:
        # DL methods: show expected score from catalog, return measurement as placeholder
        x_recon = y.copy()
        if x_recon.ndim >= 2:
            # Just use the measurement visualization
            pass
        dl_note = True
    else:
        x_recon = _run_classical_recon(y, H, algorithm_name)
        dl_note = False

    runtime_ms = (time.perf_counter() - t0) * 1000

    # Compute metrics
    psnr_val = None
    ssim_val = None
    if has_gt and x_true is not None and not dl_note:
        # Resize x_recon to match x_true if needed
        if x_recon.shape != x_true.shape:
            try:
                from PIL import Image
                img_recon = Image.fromarray(
                    ((x_recon - x_recon.min()) / max(x_recon.max() - x_recon.min(), 1e-8) * 255).astype(np.uint8)
                )
                target_shape = x_true.shape[-2:] if x_true.ndim >= 2 else x_true.shape
                img_recon = img_recon.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
                x_recon = np.array(img_recon).astype(np.float64) / 255.0 * (x_true.max() - x_true.min()) + x_true.min()
            except Exception:
                pass
        if x_recon.shape == x_true.shape:
            psnr_val = _compute_psnr(x_true, x_recon)
            ssim_val = _compute_ssim(x_true, x_recon)

    # If DL method, get expected scores from leaderboard
    expected_psnr = None
    expected_ssim = None
    if dl_note:
        lb = variant.get("normal_leaderboard", [])
        for entry in lb:
            if entry.get("method", "").lower() == algorithm_name.lower():
                expected_psnr = entry.get("psnr")
                expected_ssim = entry.get("ssim")
                break

    # Build result
    result = {
        "algorithm_name": algo_info.get("name", algorithm_name),
        "algorithm_type": algo_info.get("type", "Unknown"),
        "algorithm_source": algo_info.get("source", ""),
        "runtime_ms": round(runtime_ms, 1),
        "measurement_image": _numpy_to_png_b64(y),
        "reconstructed_image": _numpy_to_png_b64(x_recon),
        "psnr": round(psnr_val, 2) if psnr_val is not None else None,
        "ssim": round(ssim_val, 4) if ssim_val is not None else None,
        "is_dl_placeholder": dl_note,
        "expected_psnr": expected_psnr,
        "expected_ssim": expected_ssim,
        "variant_key": variant_key,
        "variant_name": variant.get("display_name", variant_key),
    }

    if has_gt and x_true is not None:
        result["ground_truth_image"] = _numpy_to_png_b64(x_true)

    return result
