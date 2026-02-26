#!/usr/bin/env python3
"""Pre-compute benchmark gallery results for CASSI, CACTI, and SPC variants.

Usage:
    python3 scripts/precompute_benchmark_results.py --all
    python3 scripts/precompute_benchmark_results.py --variant sd_cassi
    python3 scripts/precompute_benchmark_results.py --variant cacti
    python3 scripts/precompute_benchmark_results.py --variant spc_block

Outputs:
    pwm_platform/static/img/benchmark_gallery/{variant}/scene_{nn}/
        gt.png, measurement.png, recon_I.png, recon_II.png, recon_III.png
    pwm_platform/static/benchmark-data/benchmark_gallery.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Ensure pwm_core is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "packages" / "pwm_core"))

from pwm_core.mismatch.operators import (
    cacti_forward,
    cacti_gap_tv,
    cassi_forward,
    cassi_gap_denoise,
    compute_psnr,
    spc_forward,
    spc_lsq_recon,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_grayscale_png(arr: np.ndarray, path: str):
    """Save a 2D float32 array as an 8-bit grayscale PNG."""
    from PIL import Image
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img.save(path)


def _save_rgb_png(arr: np.ndarray, path: str):
    """Save an (H,W,3) float32 array as an 8-bit RGB PNG."""
    from PIL import Image
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
    img.save(path)


def _hsi_to_rgb(cube: np.ndarray) -> np.ndarray:
    """Convert hyperspectral cube (H,W,nB) to pseudo-RGB by picking 3 bands."""
    nB = cube.shape[2]
    r_idx = min(nB - 1, int(nB * 0.75))
    g_idx = min(nB - 1, int(nB * 0.50))
    b_idx = min(nB - 1, int(nB * 0.25))
    rgb = np.stack([cube[:, :, r_idx], cube[:, :, g_idx], cube[:, :, b_idx]], axis=2)
    mx = rgb.max()
    if mx > 0:
        rgb = rgb / mx
    return np.clip(rgb, 0, 1).astype(np.float32)


def _compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Simple SSIM between two 2D or 3D arrays."""
    try:
        from skimage.metrics import structural_similarity
        data_range = max(x.max(), y.max(), 1.0) - min(x.min(), y.min(), 0.0)
        if x.ndim == 3:
            val = float(structural_similarity(x, y, data_range=data_range, channel_axis=2))
        else:
            val = float(structural_similarity(x, y, data_range=data_range))
        return max(val, 0.0)
    except ImportError:
        # Fallback: simple correlation-based approximation
        x_f = x.flatten().astype(np.float64)
        y_f = y.flatten().astype(np.float64)
        mx, my = x_f.mean(), y_f.mean()
        sx, sy = x_f.std(), y_f.std()
        if sx < 1e-10 or sy < 1e-10:
            return 1.0 if np.allclose(x_f, y_f) else 0.0
        cov = np.mean((x_f - mx) * (y_f - my))
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mx * my + c1) * (2 * cov + c2)) / \
               ((mx ** 2 + my ** 2 + c1) * (sx ** 2 + sy ** 2 + c2))
        return float(np.clip(ssim, 0, 1))


# ---------------------------------------------------------------------------
# CASSI benchmark
# ---------------------------------------------------------------------------

def run_cassi_benchmark(out_root: Path, recon_iters: int = 60):
    """Run 3-scenario benchmark on all KAIST synthetic scenes."""
    from pwm_core.data.loaders.kaist import KAISTDataset

    print("[CASSI] Loading 10 KAIST synthetic scenes...")
    ds = KAISTDataset(resolution=256, num_bands=28)
    assert len(ds) == 10, f"Expected 10 scenes, got {len(ds)}"

    rng = np.random.RandomState(42)
    results = []

    for idx, (name, cube) in enumerate(ds):
        print(f"  Scene {idx:02d}/{len(ds)}: {name}")
        scene_dir = out_root / "sd_cassi" / f"scene_{idx:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        h, w, nC = cube.shape
        mask = (rng.rand(h, w) > 0.5).astype(np.float32)

        # Save ground truth (pseudo-RGB)
        gt_rgb = _hsi_to_rgb(cube)
        _save_rgb_png(gt_rgb, str(scene_dir / "gt.png"))

        # --- Scenario I: Ideal (step=2, reconstruct with step=2) ---
        nominal_step = 2
        y_ideal = cassi_forward(cube, mask, step=nominal_step)
        y_norm = y_ideal / (y_ideal.max() + 1e-8)
        _save_grayscale_png(y_norm, str(scene_dir / "measurement_I.png"))
        x_hat_I = cassi_gap_denoise(y_ideal, mask, step=nominal_step, max_iter=recon_iters)
        _save_rgb_png(_hsi_to_rgb(x_hat_I), str(scene_dir / "recon_I.png"))
        psnr_I = compute_psnr(cube, x_hat_I)
        ssim_I = _compute_ssim(cube, x_hat_I)

        # --- Scenario II: Mismatch (forward with step=3, reconstruct with step=2) ---
        perturbed_step = 3
        y_mismatch = cassi_forward(cube, mask, step=perturbed_step)
        y_norm2 = y_mismatch / (y_mismatch.max() + 1e-8)
        _save_grayscale_png(y_norm2, str(scene_dir / "measurement_II.png"))
        x_hat_II = cassi_gap_denoise(y_mismatch, mask, step=nominal_step, max_iter=recon_iters)
        _save_rgb_png(_hsi_to_rgb(x_hat_II), str(scene_dir / "recon_II.png"))
        psnr_II = compute_psnr(cube, x_hat_II)
        ssim_II = _compute_ssim(cube, x_hat_II)

        # --- Scenario III: Oracle (forward with step=3, reconstruct with step=3) ---
        x_hat_III = cassi_gap_denoise(y_mismatch, mask, step=perturbed_step, max_iter=recon_iters)
        _save_rgb_png(_hsi_to_rgb(x_hat_III), str(scene_dir / "recon_III.png"))
        psnr_III = compute_psnr(cube, x_hat_III)
        ssim_III = _compute_ssim(cube, x_hat_III)

        results.append({
            "scene_idx": idx,
            "scene_name": name,
            "psnr_I": round(psnr_I, 2),
            "ssim_I": round(ssim_I, 4),
            "psnr_II": round(psnr_II, 2),
            "ssim_II": round(ssim_II, 4),
            "psnr_III": round(psnr_III, 2),
            "ssim_III": round(ssim_III, 4),
        })
        print(f"    I={psnr_I:.2f} dB  II={psnr_II:.2f} dB  III={psnr_III:.2f} dB")

    return {
        "variant": "sd_cassi",
        "method": "GAP-Denoise",
        "mismatch_param": "dispersion_step",
        "nominal": 2,
        "perturbed": 3,
        "num_scenes": len(results),
        "scenes": results,
        "mean_psnr_I": round(np.mean([r["psnr_I"] for r in results]), 2),
        "mean_psnr_II": round(np.mean([r["psnr_II"] for r in results]), 2),
        "mean_psnr_III": round(np.mean([r["psnr_III"] for r in results]), 2),
        "mean_ssim_I": round(np.mean([r["ssim_I"] for r in results]), 4),
        "mean_ssim_II": round(np.mean([r["ssim_II"] for r in results]), 4),
        "mean_ssim_III": round(np.mean([r["ssim_III"] for r in results]), 4),
    }


# ---------------------------------------------------------------------------
# CACTI benchmark
# ---------------------------------------------------------------------------

def run_cacti_benchmark(out_root: Path, recon_iters: int = 60):
    """Run 3-scenario benchmark on 6 synthetic CACTI videos."""
    from pwm_core.data.loaders.cacti_bench import SyntheticCACTIDataset

    print("[CACTI] Loading 6 synthetic video sequences...")
    ds = SyntheticCACTIDataset(resolution=256, num_frames=8)
    assert len(ds) == 6, f"Expected 6 videos, got {len(ds)}"

    results = []

    for idx, (name, video, masks) in enumerate(ds):
        print(f"  Video {idx:02d}/{len(ds)}: {name}")
        scene_dir = out_root / "cacti" / f"scene_{idx:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Save ground truth (middle frame)
        mid = video.shape[2] // 2
        _save_grayscale_png(video[:, :, mid], str(scene_dir / "gt.png"))

        # --- Scenario I: Ideal (timing_offset=0) ---
        y_ideal = cacti_forward(video, masks, timing_offset=0)
        y_norm = y_ideal / (y_ideal.max() + 1e-8)
        _save_grayscale_png(y_norm, str(scene_dir / "measurement_I.png"))
        x_hat_I = cacti_gap_tv(y_ideal, masks, timing_offset=0, iters=recon_iters)
        _save_grayscale_png(x_hat_I[:, :, mid], str(scene_dir / "recon_I.png"))
        psnr_I = compute_psnr(video, x_hat_I)
        ssim_I = _compute_ssim(video[:, :, mid], x_hat_I[:, :, mid])

        # --- Scenario II: Mismatch (forward with offset=2, reconstruct with offset=0) ---
        y_mismatch = cacti_forward(video, masks, timing_offset=2)
        y_norm2 = y_mismatch / (y_mismatch.max() + 1e-8)
        _save_grayscale_png(y_norm2, str(scene_dir / "measurement_II.png"))
        x_hat_II = cacti_gap_tv(y_mismatch, masks, timing_offset=0, iters=recon_iters)
        _save_grayscale_png(x_hat_II[:, :, mid], str(scene_dir / "recon_II.png"))
        psnr_II = compute_psnr(video, x_hat_II)
        ssim_II = _compute_ssim(video[:, :, mid], x_hat_II[:, :, mid])

        # --- Scenario III: Oracle (forward with offset=2, reconstruct with offset=2) ---
        x_hat_III = cacti_gap_tv(y_mismatch, masks, timing_offset=2, iters=recon_iters)
        _save_grayscale_png(x_hat_III[:, :, mid], str(scene_dir / "recon_III.png"))
        psnr_III = compute_psnr(video, x_hat_III)
        ssim_III = _compute_ssim(video[:, :, mid], x_hat_III[:, :, mid])

        results.append({
            "scene_idx": idx,
            "scene_name": name,
            "psnr_I": round(psnr_I, 2),
            "ssim_I": round(ssim_I, 4),
            "psnr_II": round(psnr_II, 2),
            "ssim_II": round(ssim_II, 4),
            "psnr_III": round(psnr_III, 2),
            "ssim_III": round(ssim_III, 4),
        })
        print(f"    I={psnr_I:.2f} dB  II={psnr_II:.2f} dB  III={psnr_III:.2f} dB")

    return {
        "variant": "cacti",
        "method": "GAP-TV",
        "mismatch_param": "timing_offset",
        "nominal": 0,
        "perturbed": 2,
        "num_scenes": len(results),
        "scenes": results,
        "mean_psnr_I": round(np.mean([r["psnr_I"] for r in results]), 2),
        "mean_psnr_II": round(np.mean([r["psnr_II"] for r in results]), 2),
        "mean_psnr_III": round(np.mean([r["psnr_III"] for r in results]), 2),
        "mean_ssim_I": round(np.mean([r["ssim_I"] for r in results]), 4),
        "mean_ssim_II": round(np.mean([r["ssim_II"] for r in results]), 4),
        "mean_ssim_III": round(np.mean([r["ssim_III"] for r in results]), 4),
    }


# ---------------------------------------------------------------------------
# SPC benchmark
# ---------------------------------------------------------------------------

def run_spc_benchmark(out_root: Path):
    """Run 3-scenario benchmark on 11 Set11 synthetic images."""
    from pwm_core.data.loaders.set11 import Set11Dataset

    print("[SPC] Loading 11 Set11 synthetic images...")
    ds = Set11Dataset(resolution=33)
    assert len(ds) == 11, f"Expected 11 images, got {len(ds)}"

    rng = np.random.RandomState(99)
    cs_ratio = 0.25
    results = []

    for idx, (name, img) in enumerate(ds):
        print(f"  Image {idx:02d}/{len(ds)}: {name}")
        scene_dir = out_root / "spc_block" / f"scene_{idx:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        _save_grayscale_png(img, str(scene_dir / "gt.png"))

        n = img.shape[0]
        n_pix = n * n
        n_meas = int(cs_ratio * n_pix)
        Phi = rng.randn(n_meas, n_pix).astype(np.float32) / np.sqrt(n_meas)
        x_flat = img.flatten()

        # --- Scenario I: Ideal (gain=1.0, bias=0.0) ---
        y_ideal = spc_forward(x_flat, Phi, gain=1.0, bias=0.0)
        x_hat_I = spc_lsq_recon(y_ideal, Phi, gain=1.0, bias=0.0)
        recon_I = x_hat_I.reshape(n, n)
        _save_grayscale_png(recon_I, str(scene_dir / "recon_I.png"))
        psnr_I = compute_psnr(img, recon_I)
        ssim_I = _compute_ssim(img, recon_I)

        # Save measurement as 1D bar (resize to square for display)
        meas_vis = np.zeros((n, n), dtype=np.float32)
        meas_norm = y_ideal / (np.abs(y_ideal).max() + 1e-8) * 0.5 + 0.5
        row_len = min(n_meas, n * n)
        meas_vis.flat[:row_len] = meas_norm[:row_len]
        _save_grayscale_png(meas_vis, str(scene_dir / "measurement_I.png"))

        # --- Scenario II: Mismatch (forward gain=0.8, bias=0.05; reconstruct with gain=1.0, bias=0.0) ---
        y_mismatch = spc_forward(x_flat, Phi, gain=0.8, bias=0.05)
        x_hat_II = spc_lsq_recon(y_mismatch, Phi, gain=1.0, bias=0.0)
        recon_II = x_hat_II.reshape(n, n)
        _save_grayscale_png(recon_II, str(scene_dir / "recon_II.png"))
        psnr_II = compute_psnr(img, recon_II)
        ssim_II = _compute_ssim(img, recon_II)

        meas_vis2 = np.zeros((n, n), dtype=np.float32)
        meas_norm2 = y_mismatch / (np.abs(y_mismatch).max() + 1e-8) * 0.5 + 0.5
        meas_vis2.flat[:row_len] = meas_norm2[:row_len]
        _save_grayscale_png(meas_vis2, str(scene_dir / "measurement_II.png"))

        # --- Scenario III: Oracle (forward gain=0.8, bias=0.05; reconstruct with gain=0.8, bias=0.05) ---
        x_hat_III = spc_lsq_recon(y_mismatch, Phi, gain=0.8, bias=0.05)
        recon_III = x_hat_III.reshape(n, n)
        _save_grayscale_png(recon_III, str(scene_dir / "recon_III.png"))
        psnr_III = compute_psnr(img, recon_III)
        ssim_III = _compute_ssim(img, recon_III)

        results.append({
            "scene_idx": idx,
            "scene_name": name,
            "psnr_I": round(psnr_I, 2),
            "ssim_I": round(ssim_I, 4),
            "psnr_II": round(psnr_II, 2),
            "ssim_II": round(ssim_II, 4),
            "psnr_III": round(psnr_III, 2),
            "ssim_III": round(ssim_III, 4),
        })
        print(f"    I={psnr_I:.2f} dB  II={psnr_II:.2f} dB  III={psnr_III:.2f} dB")

    return {
        "variant": "spc_block",
        "method": "Reg. Least-Squares",
        "mismatch_param": "gain+bias",
        "nominal": {"gain": 1.0, "bias": 0.0},
        "perturbed": {"gain": 0.8, "bias": 0.05},
        "num_scenes": len(results),
        "scenes": results,
        "mean_psnr_I": round(np.mean([r["psnr_I"] for r in results]), 2),
        "mean_psnr_II": round(np.mean([r["psnr_II"] for r in results]), 2),
        "mean_psnr_III": round(np.mean([r["psnr_III"] for r in results]), 2),
        "mean_ssim_I": round(np.mean([r["ssim_I"] for r in results]), 4),
        "mean_ssim_II": round(np.mean([r["ssim_II"] for r in results]), 4),
        "mean_ssim_III": round(np.mean([r["ssim_III"] for r in results]), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-compute benchmark gallery results")
    parser.add_argument("--all", action="store_true", help="Run all variants")
    parser.add_argument("--variant", type=str, help="Run a specific variant (sd_cassi, cacti, spc_block)")
    parser.add_argument("--iters", type=int, default=60, help="Reconstruction iterations")
    parser.add_argument("--upload-gcs", action="store_true", help="Upload results to GCS after computation")
    parser.add_argument("--gcs-bucket", type=str, default="pwm-benchmark-datasets", help="GCS bucket name")
    args = parser.parse_args()

    if not args.all and not args.variant:
        parser.print_help()
        sys.exit(1)

    platform_root = Path(__file__).resolve().parent.parent
    img_root = platform_root / "pwm_platform" / "static" / "img" / "benchmark_gallery"
    json_dir = platform_root / "pwm_platform" / "static" / "benchmark-data"
    json_dir.mkdir(parents=True, exist_ok=True)

    variants_to_run = []
    if args.all:
        variants_to_run = ["sd_cassi", "cacti", "spc_block"]
    else:
        variants_to_run = [args.variant]

    gallery = {}

    # Load existing gallery if present
    json_path = json_dir / "benchmark_gallery.json"
    if json_path.exists():
        with open(json_path) as f:
            gallery = json.load(f)

    for variant in variants_to_run:
        print(f"\n{'='*60}")
        print(f"  Running {variant} benchmark")
        print(f"{'='*60}\n")

        if variant == "sd_cassi":
            gallery["sd_cassi"] = run_cassi_benchmark(img_root, recon_iters=args.iters)
        elif variant == "cacti":
            gallery["cacti"] = run_cacti_benchmark(img_root, recon_iters=args.iters)
        elif variant == "spc_block":
            gallery["spc_block"] = run_spc_benchmark(img_root)
        else:
            print(f"Unknown variant: {variant}")
            continue

    # Save combined JSON
    with open(json_path, "w") as f:
        json.dump(gallery, f, indent=2)
    print(f"\nSaved gallery JSON to {json_path}")

    # Upload to GCS if requested
    if args.upload_gcs:
        _upload_to_gcs(platform_root, args.gcs_bucket)

    print("Done.")


def _upload_to_gcs(platform_root: Path, bucket_name: str):
    """Upload all gallery files to Google Cloud Storage."""
    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        print("\n[GCS] google-cloud-storage not installed, skipping upload.")
        return

    static_root = platform_root / "pwm_platform" / "static"
    img_root = static_root / "img" / "benchmark_gallery"
    json_path = static_root / "benchmark-data" / "benchmark_gallery.json"

    files = []
    if img_root.exists():
        for f in sorted(img_root.rglob("*")):
            if f.is_file():
                rel = f.relative_to(static_root)
                files.append((f, f"benchmark_gallery/{rel}"))
    if json_path.exists():
        files.append((json_path, "benchmark_gallery/benchmark_gallery.json"))

    if not files:
        print("\n[GCS] No files to upload.")
        return

    print(f"\n[GCS] Uploading {len(files)} files to gs://{bucket_name}/benchmark_gallery/ ...")
    try:
        client = gcs_storage.Client()
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            bucket = client.create_bucket(bucket_name, location="us-central1")

        uploaded = 0
        for local_path, gcs_key in files:
            blob = bucket.blob(gcs_key)
            blob.upload_from_filename(str(local_path))
            uploaded += 1
        print(f"[GCS] Uploaded {uploaded}/{len(files)} files.")
    except Exception as e:
        print(f"[GCS] Upload failed: {e}")
        print("[GCS] You can retry later with: python3 scripts/upload_gallery_to_gcs.py")


if __name__ == "__main__":
    main()
