#!/usr/bin/env python3
"""Generic Benchmark Runner — GPU Methods on Modal (T4 GPU).

Runs benchmark algorithms for ANY modality on Modal T4 GPU.
Downloads datasets from GCS, runs reconstruction, uploads results to GCS.

Usage:
    modal run scripts/modal_run_benchmark.py --variant ct
    modal run scripts/modal_run_benchmark.py --variant mri --tier public
    modal run scripts/modal_run_benchmark.py --variant pet --tier all

Prerequisites:
    modal setup
    pip install google-cloud-storage

Output:
    results/<variant>/benchmark_gpu_<timestamp>.json  (local)
    gs://pwm-benchmark-datasets/benchmark-results/<variant>/...  (GCS)
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import modal

# ── Modal setup ───────────────────────────────────────────────────────────────

app = modal.App("pwm-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "h5py",
        "scikit-image",
    )
)

# ══════════════════════════════════════════════════════════════════════════════
# Generic reconstruction algorithms (work for any modality)
# ══════════════════════════════════════════════════════════════════════════════


def _compute_psnr(x_hat, x_true):
    import numpy as np
    mse = float(np.mean((x_hat - x_true) ** 2))
    if mse < 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def _compute_ssim(x_hat, x_true):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        x_hat.astype("float32"), x_true.astype("float32"), data_range=1.0
    ))


def _composite_score(psnr, ssim, cons):
    psnr_norm = min(1.0, (psnr - 10.0) / 40.0)
    return 0.4 * psnr_norm + 0.4 * ssim + 0.2 * cons


def _tikhonov_recon(y, H, lam=1e-3):
    """Tikhonov regularized reconstruction: x = (H^T H + λI)^{-1} H^T y."""
    import numpy as np
    if H is None:
        return y  # No forward model, return measurement as-is
    HtH = H.T @ H
    Hty = H.T @ y.ravel()
    I = np.eye(HtH.shape[0])
    x_hat = np.linalg.solve(HtH + lam * I, Hty)
    return x_hat.reshape(y.shape if H is None else (int(np.sqrt(len(x_hat))),) * 2)


def _tv_denoising(x, lam=0.1, n_iter=50):
    """Simple TV denoising via Chambolle proximal gradient."""
    import numpy as np
    z = x.copy().astype(np.float64)
    for _ in range(n_iter):
        # Gradient
        dy = np.diff(z, axis=0, append=0)
        dx = np.diff(z, axis=1, append=0)
        # Divergence of normalized gradient
        mag = np.sqrt(dy**2 + dx**2 + 1e-8)
        ny, nx = dy / mag, dx / mag
        div_y = np.diff(ny, axis=0, prepend=0)
        div_x = np.diff(nx, axis=1, prepend=0)
        z = x + lam * (div_y + div_x)
    return np.clip(z, 0, 1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Modal remote function
# ══════════════════════════════════════════════════════════════════════════════


@app.function(
    image=image,
    gpu="T4",    # T4 GPU — user requirement; do NOT change to A10/A100
    timeout=7200,
    memory=16384,
)
def run_tier_gpu(h5_bytes: bytes, variant: str, tier: str) -> list[dict]:
    """Process one tier of any modality on T4 GPU.

    Returns list of per-scene metric dicts.
    """
    import torch
    import h5py
    import numpy as np
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"
    print(f"[{variant}/{tier}] Device: {device}  GPU: {gpu_name}")

    rows = []
    f = h5py.File(io.BytesIO(h5_bytes), "r")
    sample_keys = sorted([k for k in f.keys() if k.startswith("sample_")])
    if not sample_keys:
        sample_keys = sorted(f.keys())

    has_gt = False
    for sk in sample_keys:
        if "x_true" in f[sk]:
            has_gt = True
            break

    print(f"  Samples: {len(sample_keys)}, has_gt={has_gt}")

    for sk in sample_keys:
        grp = f[sk]
        meta = {}
        if "metadata" in grp.attrs:
            try:
                meta = json.loads(grp.attrs["metadata"])
            except Exception:
                pass

        # Load data
        x_true = grp["x_true"][()].astype(np.float64) if "x_true" in grp else None
        y = grp["y"][()].astype(np.float64) if "y" in grp else None
        H = grp["H_ideal"][()].astype(np.float64) if "H_ideal" in grp else None

        if y is None:
            # Try alternative key names
            for key in ["sinogram_measured", "measurement", "data"]:
                if key in grp:
                    y = grp[key][()].astype(np.float64)
                    break

        if y is None:
            print(f"  [{sk}] No measurement data found, skipping")
            continue

        # Normalize x_true to [0,1] if needed
        if x_true is not None:
            xt_min, xt_max = x_true.min(), x_true.max()
            if xt_max - xt_min > 1e-10:
                x_true = (x_true - xt_min) / (xt_max - xt_min)

        print(f"\n  [{tier}] {sk}  y.shape={y.shape}"
              f"  x_true={'yes' if x_true is not None else 'no'}"
              f"  H={'yes' if H is not None else 'no'}")

        # Run algorithms
        algos_to_run = ["identity", "tikhonov", "tv_denoise"]

        for algo_name in algos_to_run:
            t0 = time.time()
            try:
                if algo_name == "identity":
                    # Use measurement as reconstruction (baseline)
                    if H is not None and H.ndim == 2:
                        x_hat = np.linalg.lstsq(H, y.ravel(), rcond=None)[0]
                        n = int(np.sqrt(len(x_hat)))
                        x_hat = x_hat.reshape(n, n) if n * n == len(x_hat) else x_hat
                    else:
                        x_hat = y.copy()
                elif algo_name == "tikhonov":
                    if H is not None and H.ndim == 2:
                        x_hat = _tikhonov_recon(y, H, lam=1e-3)
                    else:
                        x_hat = y.copy()
                elif algo_name == "tv_denoise":
                    if H is not None and H.ndim == 2:
                        x_hat = _tikhonov_recon(y, H, lam=1e-4)
                        x_hat = _tv_denoising(x_hat.astype(np.float32))
                    else:
                        x_hat = _tv_denoising(y.astype(np.float32))
                else:
                    continue
            except Exception as exc:
                print(f"    [{algo_name}] ERROR: {exc}")
                continue

            elapsed = time.time() - t0

            # Normalize x_hat to [0,1]
            xh_min, xh_max = x_hat.min(), x_hat.max()
            if xh_max - xh_min > 1e-10:
                x_hat = (x_hat - xh_min) / (xh_max - xh_min)
            x_hat = np.clip(x_hat, 0, 1).astype(np.float64)

            # Compute metrics
            if x_true is not None and x_hat.shape == x_true.shape:
                psnr = _compute_psnr(x_hat, x_true)
                ssim = _compute_ssim(x_hat.astype(np.float32), x_true.astype(np.float32))
            else:
                psnr = 0.0
                ssim = 0.0

            # Consistency: ||y - H x_hat|| / ||y||
            cons = 0.0
            if H is not None and y is not None:
                try:
                    y_pred = H @ x_hat.ravel()
                    y_norm = float(np.linalg.norm(y.ravel()))
                    if y_norm > 1e-8:
                        cons = float(1.0 - np.linalg.norm(y_pred - y.ravel()) / y_norm)
                except Exception:
                    pass

            score = _composite_score(psnr, ssim, cons)

            print(f"    [{algo_name:12s}]  PSNR={psnr:6.2f} dB  SSIM={ssim:.4f}  "
                  f"Cons={cons:.4f}  Score={score:.4f}  t={elapsed:.1f}s")

            rows.append({
                "variant":     variant,
                "tier":        tier,
                "scene":       sk,
                "algo":        algo_name,
                "psnr_db":     round(psnr, 4),
                "ssim":        round(ssim, 4),
                "consistency": round(cons, 4),
                "score":       round(score, 4),
                "time_s":      round(elapsed, 2),
            })

    f.close()
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# GCS helpers
# ══════════════════════════════════════════════════════════════════════════════


GCS_BUCKET = "pwm-benchmark-datasets"


def _download_from_gcs(variant: str, tier: str) -> bytes:
    """Download challenge HDF5 from GCS bucket, return bytes."""
    from google.cloud import storage as gcs_storage

    blob_key = f"challenge-data/v1.0/{variant}_challenge_{tier}.h5"
    client = gcs_storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_key)

    if not blob.exists():
        raise FileNotFoundError(f"Not found: gs://{GCS_BUCKET}/{blob_key}")

    print(f"  Downloading gs://{GCS_BUCKET}/{blob_key} ...")
    data = blob.download_as_bytes()
    print(f"  Downloaded {len(data) // 1024} KB")
    return data


def _upload_to_gcs(local_path: Path, gcs_key: str) -> str:
    """Upload file to GCS bucket. Returns gs:// URI."""
    from google.cloud import storage as gcs_storage

    client = gcs_storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(gcs_key)
    blob.upload_from_filename(str(local_path))
    uri = f"gs://{GCS_BUCKET}/{gcs_key}"
    print(f"  Uploaded → {uri}")
    return uri


# ══════════════════════════════════════════════════════════════════════════════
# Local entrypoint
# ══════════════════════════════════════════════════════════════════════════════


@app.local_entrypoint()
def main(
    variant: str = "ct",
    tier: str = "all",
):
    """Run benchmark for any modality on Modal T4 GPU.

    Downloads datasets from GCS, runs on Modal T4, saves results locally
    and uploads to GCS.

    --variant  ct|mri|pet|...  (any modality variant key)
    --tier     public|dev|hidden|all  (default: all)
    """
    import csv
    from collections import defaultdict
    from datetime import datetime, timezone

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RESULTS_DIR = PROJECT_ROOT / "results" / variant
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ALL_TIERS = ["public", "dev", "hidden"]
    tiers = ALL_TIERS if tier == "all" else [t.strip() for t in tier.split(",")]

    print(f"Benchmark — {variant} — Modal T4 GPU")
    print(f"Tiers: {tiers}")
    print(f"Source: gs://{GCS_BUCKET}/challenge-data/v1.0/")

    # Submit all tiers in parallel
    futures = {}
    for t in tiers:
        try:
            h5_bytes = _download_from_gcs(variant, t)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        futures[t] = run_tier_gpu.spawn(h5_bytes, variant, t)

    # Collect results
    all_rows = []
    for t, future in futures.items():
        print(f"  [WAITING] {t} ...")
        rows = future.get()
        all_rows.extend(rows)
        print(f"  [DONE] {t}: {len(rows)} results")

    # Save locally
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = RESULTS_DIR / f"benchmark_gpu_{ts}.json"
    out_csv = RESULTS_DIR / f"benchmark_gpu_{ts}.csv"

    result_doc = {
        "variant": variant,
        "timestamp": ts,
        "tiers": tiers,
        "gpu": "T4",
        "source": f"gs://{GCS_BUCKET}/challenge-data/v1.0/",
        "scenes": all_rows,
    }
    with open(out_json, "w") as fj:
        json.dump(result_doc, fj, indent=2)
    print(f"\nResults saved → {out_json}")

    if all_rows:
        with open(out_csv, "w", newline="") as fc:
            writer = csv.DictWriter(fc, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"CSV saved     → {out_csv}")

    # Upload results to GCS
    try:
        gcs_json = f"benchmark-results/{variant}/benchmark_gpu_{ts}.json"
        _upload_to_gcs(out_json, gcs_json)
        if out_csv.exists():
            gcs_csv = f"benchmark-results/{variant}/benchmark_gpu_{ts}.csv"
            _upload_to_gcs(out_csv, gcs_csv)
    except Exception as e:
        print(f"[WARN] GCS upload failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY — {variant} — mean metrics per (tier, algo)")
    print("=" * 70)
    print(f"{'tier':8s}  {'algo':14s}  {'PSNR':>7s}  {'SSIM':>6s}  {'Cons':>6s}  {'Score':>6s}")
    print("-" * 70)
    acc: dict = defaultdict(list)
    for r in all_rows:
        acc[(r["tier"], r["algo"])].append(r)
    for (t, a), rows in sorted(acc.items()):
        psnr = sum(r["psnr_db"] for r in rows) / len(rows)
        ssim = sum(r["ssim"] for r in rows) / len(rows)
        cons = sum(r["consistency"] for r in rows) / len(rows)
        score = sum(r["score"] for r in rows) / len(rows)
        print(f"{t:8s}  {a:14s}  {psnr:7.2f}  {ssim:6.4f}  {cons:6.4f}  {score:6.4f}")
    print("=" * 70)
