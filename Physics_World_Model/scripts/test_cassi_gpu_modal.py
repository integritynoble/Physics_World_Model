#!/usr/bin/env python3
"""Test all CASSI GPU algorithms on Modal T4.

Tests each GPU algorithm ONCE on the CASSI standard dataset (standard_cassi_00.h5).
Downloads checkpoints locally using gsutil, passes bytes to Modal T4.

Usage:
    modal run scripts/test_cassi_gpu_modal.py
    modal run scripts/test_cassi_gpu_modal.py --model-key dauhst_9stg
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
CASSI_ARCH_DIR = REPO_ROOT / "packages" / "pwm_core" / "pwm_core" / "recon" / "cassi_arch"
RECON_DIR = REPO_ROOT / "packages" / "pwm_core" / "pwm_core" / "recon"

app = modal.App("pwm-cassi-gpu-test-v3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "numpy<2.0",
        "scipy",
        "h5py",
        "scikit-image",
        "einops",
        "timm",
    )
    .add_local_dir(str(CASSI_ARCH_DIR), "/cassi_arch")
    .add_local_file(str(RECON_DIR / "cassi_models.py"), "/cassi_models.py")
    .add_local_file(str(RECON_DIR / "mst.py"), "/mst.py")
)

GCS_BUCKET = "pwm-benchmark-datasets"

# Standard dataset path (canonical 3D KAIST format)
STANDARD_DATASET_GCS = "datasets/Benchmark/cassi/standard/standard_cassi_00.h5"

# GPU algorithms with checkpoints available in GCS checkpoint/cassi/
# model_type: "cassi_models" uses cassi_models.py
GPU_ALGORITHMS = {
    "tsa_net":        {"display": "TSA-Net",      "ckpt": "checkpoint/cassi/tsa_net/tsa_net.pth",             "ref_psnr": 31.5,  "type": "cassi_models", "model_key": "tsa_net"},
    "dgsmp":          {"display": "DGSMP",         "ckpt": "checkpoint/cassi/dgsmp/dgsmp.pth",                 "ref_psnr": 32.6,  "type": "cassi_models", "model_key": "dgsmp"},
    "lambda_net":     {"display": "λ-Net",          "ckpt": "checkpoint/cassi/lambda_net/lambda_net.pth",       "ref_psnr": 30.1,  "type": "cassi_models", "model_key": "lambda_net"},
    "admm_net":       {"display": "ADMM-Net",       "ckpt": "checkpoint/cassi/admm_net/admm_net.pth",           "ref_psnr": 29.1,  "type": "cassi_models", "model_key": "admm_net"},
    "gap_net":        {"display": "GAP-Net",        "ckpt": "checkpoint/cassi/gap_net/gap_net.pth",             "ref_psnr": 29.1,  "type": "cassi_models", "model_key": "gap_net"},
    "hdnet":          {"display": "HDNet",          "ckpt": "checkpoint/cassi/hdnet/hdnet.pth",                 "ref_psnr": 34.97, "type": "cassi_models", "model_key": "hdnet"},
    "mst_l":          {"display": "MST-L",          "ckpt": "checkpoint/cassi/mst/mst_l.pth",                  "ref_psnr": 34.81, "type": "cassi_models", "model_key": "mst_l"},
    "mst_plus_plus":  {"display": "MST++",          "ckpt": "checkpoint/cassi/mst_plus_plus/mst_plus_plus.pth", "ref_psnr": 36.0,  "type": "cassi_models", "model_key": "mst_plus_plus"},
    "cst_l_plus":     {"display": "CST-L-Plus",     "ckpt": "checkpoint/cassi/cst/cst_l_plus.pth",             "ref_psnr": 36.1,  "type": "cassi_models", "model_key": "cst_l_plus"},
    "birnat":         {"display": "BIRNAT",         "ckpt": "checkpoint/cassi/birnat/birnat.pth",               "ref_psnr": 30.0,  "type": "cassi_models", "model_key": "birnat"},
    "dauhst_9stg":    {"display": "DAUHST-9stg",   "ckpt": "checkpoint/cassi/dauhst/dauhst_9stg.pth",          "ref_psnr": 38.4,  "type": "cassi_models", "model_key": "dauhst_9stg"},
    "bisrnet":        {"display": "BiSRNet",        "ckpt": "checkpoint/cassi/bisrnet/bisrnet.pth",             "ref_psnr": 33.0,  "type": "cassi_models", "model_key": "bisrnet"},
    "padut_3stg":     {"display": "PADUT-3stg",     "ckpt": "checkpoint/cassi/padut/padut_3stg_official.pth",   "ref_psnr": 36.95, "type": "cassi_models", "model_key": "padut_3stg"},
    "rdluf_mixs2_9stg": {"display": "RDLUF-MixS2-9stg", "ckpt": "checkpoint/cassi/rdluf_mixs2/RDLUF_MixS2_9stage.pth", "ref_psnr": 39.6, "type": "cassi_models", "model_key": "rdluf_mixs2_9stg"},
    "ssr_l":          {"display": "SSR-L",          "ckpt": "checkpoint/cassi/ssr/model_L.pkl",                 "ref_psnr": 34.0,  "type": "cassi_models", "model_key": "ssr_l"},
}

CKPT_CACHE_DIR = Path("/tmp/cassi_ckpts")


def _download_gcs_file(gcs_path: str, local_path: Path) -> bool:
    """Download a file from GCS using gsutil. Returns True if successful."""
    if local_path.exists():
        return True
    local_path.parent.mkdir(parents=True, exist_ok=True)
    gcs_url = f"gs://{GCS_BUCKET}/{gcs_path}"
    result = subprocess.run(
        ["gsutil", "cp", gcs_url, str(local_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR downloading {gcs_url}: {result.stderr[:200]}")
        return False
    return True


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    memory=16384,
)
def run_cassi_inference(
    model_key: str,
    algo_info: dict,
    ckpt_bytes: bytes,
    h5_bytes: bytes,
) -> dict:
    """Run inference for one CASSI model on T4 GPU."""
    import sys
    import time
    import tempfile

    import numpy as np
    import h5py
    import torch

    display = algo_info["display"]
    ref_psnr = algo_info["ref_psnr"]
    cassi_model_key = algo_info["model_key"]

    # Add cassi_arch to path (mounted at /cassi_arch as a package dir)
    sys.path.insert(0, "/")  # makes /cassi_arch importable as 'cassi_arch'

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[{model_key}] GPU: {device_name}, torch={torch.__version__}, numpy={np.__version__}", flush=True)

    # Write checkpoint to disk
    ckpt_path = f"/tmp/ckpt_{model_key}.pth"
    with open(ckpt_path, "wb") as f:
        f.write(ckpt_bytes)
    print(f"[{model_key}] Checkpoint written: {len(ckpt_bytes)//1024//1024} MB", flush=True)

    # Load standard dataset (keys: x_true, y_ideal, mask, wavelength)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
        tf.write(h5_bytes)
        tf_name = tf.name
    with h5py.File(tf_name, "r") as hf:
        x_true = hf["x_true"][:].astype(np.float32)   # (256, 256, 28)
        y_ideal = hf["y_ideal"][:].astype(np.float32)  # (256, 310)
        mask = hf["mask"][:].astype(np.float32)         # (256, 256)

    nC = x_true.shape[2]
    step = 2
    print(f"[{model_key}] Data: x_true={x_true.shape}, y_ideal={y_ideal.shape}, mask={mask.shape}", flush=True)

    def compute_psnr(x_hat, x_true):
        """Per-band average PSNR with peak=1 (standard CASSI benchmark formula)."""
        nC = x_true.shape[-1]
        psnr_total = 0.0
        x_hat_f = x_hat.astype(np.float64)
        x_true_f = x_true.astype(np.float64)
        for i in range(nC):
            mse = np.mean((x_hat_f[:, :, i] - x_true_f[:, :, i]) ** 2)
            psnr_total += 10.0 * np.log10(1.0 / (mse + 1e-8))
        return psnr_total / nC

    try:
        t0 = time.time()

        # Use cassi_models.py for all models
        import importlib.util
        spec_mod = importlib.util.spec_from_file_location("cassi_models", "/cassi_models.py")
        cassi_models_mod = importlib.util.module_from_spec(spec_mod)
        sys.modules["cassi_models"] = cassi_models_mod
        spec_mod.loader.exec_module(cassi_models_mod)

        x_hat = cassi_models_mod.cassi_model_recon(
            y=y_ideal,
            mask_2d=mask,
            model_key=cassi_model_key,
            nC=nC,
            step=step,
            ckpt_path=ckpt_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            x_true=x_true,
        )

        dt = time.time() - t0
        x_hat = np.asarray(x_hat, dtype=np.float32)
        psnr = compute_psnr(x_hat, x_true)
        passed = psnr >= ref_psnr * 0.80
        status = "PASS" if passed else "LOW_PSNR"
        print(f"[{model_key}] PSNR: {psnr:.2f} dB / ref {ref_psnr:.1f} dB  Time: {dt:.1f}s → {status}", flush=True)
        return {
            "status": status,
            "psnr": round(psnr, 2),
            "ref_psnr": ref_psnr,
            "time_s": round(dt, 1),
            "display": display,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        short_err = str(e)[:250]
        print(f"[{model_key}] FAILED: {short_err}", flush=True)
        print(tb[:800], flush=True)
        return {
            "status": "FAIL",
            "error": short_err,
            "psnr": None,
            "display": display,
        }


@app.local_entrypoint()
def main(model_key: str = "all"):
    """Run CASSI GPU tests."""
    import json
    import datetime

    CKPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if model_key == "all":
        keys_to_test = list(GPU_ALGORITHMS.keys())
    else:
        keys_to_test = [model_key]

    print(f"Testing {len(keys_to_test)} CASSI GPU algorithms on Modal T4")

    # Download standard dataset locally
    h5_local = CKPT_CACHE_DIR / "standard_cassi_00.h5"
    if not h5_local.exists():
        print("Downloading standard CASSI dataset from GCS...")
        ok = _download_gcs_file(STANDARD_DATASET_GCS, h5_local)
        if not ok:
            print("ERROR: Cannot get standard CASSI dataset")
            return {}

    h5_bytes = h5_local.read_bytes()
    print(f"Dataset loaded: {len(h5_bytes)//1024//1024} MB")

    results = {}

    for mk in keys_to_test:
        if mk not in GPU_ALGORITHMS:
            print(f"[{mk}] Unknown model key, skipping")
            continue

        info = GPU_ALGORITHMS[mk]
        display = info["display"]

        print(f"\n--- {display} ({mk}) ---")

        # Download checkpoint locally
        ckpt_local = CKPT_CACHE_DIR / f"{mk}.pth"
        ok = _download_gcs_file(info["ckpt"], ckpt_local)
        if not ok:
            results[mk] = {"status": "FAIL", "error": "checkpoint not downloaded", "psnr": None, "display": display}
            continue

        ckpt_bytes = ckpt_local.read_bytes()
        print(f"Checkpoint: {len(ckpt_bytes)//1024//1024} MB")

        # Run on Modal T4 (one time only per algorithm)
        result = run_cassi_inference.remote(
            model_key=mk,
            algo_info=info,
            ckpt_bytes=ckpt_bytes,
            h5_bytes=h5_bytes,
        )
        results[mk] = result
        status = result.get("status", "UNKNOWN")
        psnr = result.get("psnr")
        psnr_str = f"{psnr:.2f} dB" if psnr is not None else "N/A"
        print(f"  Result: {status}, PSNR={psnr_str}")

    # Summary
    print("\n" + "=" * 75)
    print("CASSI GPU ALGORITHM TEST RESULTS (Modal T4)")
    print("=" * 75)
    print(f"{'Model Key':<20} {'Display':<18} {'PSNR':>9} {'Ref':>7} {'Time':>7} Status")
    print("-" * 75)

    passed_count = 0
    for mk, res in results.items():
        info = GPU_ALGORITHMS.get(mk, {})
        display = res.get("display", mk)
        psnr = res.get("psnr")
        ref = res.get("ref_psnr", info.get("ref_psnr", 0))
        time_s = res.get("time_s", 0)
        status = res.get("status", "UNKNOWN")
        psnr_str = f"{psnr:.2f} dB" if psnr is not None else "N/A"
        time_str = f"{time_s:.1f}s" if time_s else "N/A"
        print(f"{mk:<20} {display:<18} {psnr_str:>9} {ref:>6.1f} {time_str:>7} {status}")
        if status == "PASS":
            passed_count += 1

    print(f"\nPassed: {passed_count}/{len(results)}")

    # Save results
    out_file = Path(__file__).parent.parent / "results" / "cassi_gpu_test_results.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "test_date": datetime.date.today().isoformat(),
            "gpu": "T4",
            "dataset": "standard_cassi_00.h5 (KAIST scene 0)",
            "psnr_formula": "per-band average PSNR with peak=1",
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {out_file}")

    return results
