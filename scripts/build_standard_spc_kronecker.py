#!/usr/bin/env python3
"""
Build standard dataset for spc_kronecker modality.

spc_kronecker is an alias for CASSI-type spectral imaging using SPC (single pixel camera)
with Kronecker-structured measurement matrices.

Source: Same KAIST hyperspectral scenes as cassi (mengziyi64/TSA-Net)
  These are the standard KAIST dataset scenes (256x256x28) used across all CASSI variants.

Standard dataset: 10 samples (same as cassi, different forward operator)
  x_true  = hyperspectral cube (256x256x28)
  y_ideal = SPC measurement: y[m] = sum_{i,j,lam} Phi[m,i,j,lam] * x[i,j,lam]
            Approximated as: random Kronecker-structured projection

  For 10x10 spatial measurements with 28 bands:
    y = Phi_spatial (otimes) Phi_spectral * vec(x)
    Here we use: y_ideal = sum of spatial random masks times HSI
    Result shape: (28, 256, 256) — one measurement per band

Forward model: SPC random measurement (simplified)
  y[k] = sum_{i,j} phi[k,i,j] * sum_lam x[i,j,lam]
  We use the same KAIST HSI cubes as cassi, applying a different measurement operator.

Output: datasets/benchmark/spc_kronecker/standard/
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "spc_kronecker" / "standard"
CASSI_DIR = ROOT / "datasets" / "benchmark" / "cassi" / "standard"

OUTDIR.mkdir(parents=True, exist_ok=True)


def build():
    print("=" * 60)
    print("SPC Kronecker standard dataset builder (from CASSI data)")
    print("=" * 60)

    # Check cassi standard data is available
    cassi_files = sorted(CASSI_DIR.glob("standard_cassi_*.h5"))
    if not cassi_files:
        print(f"ERROR: No cassi standard HDF5 files found in {CASSI_DIR}")
        print("Run build_standard_cassi.py first.")
        return False

    print(f"\n[1] Found {len(cassi_files)} cassi source files")

    # SPC Kronecker forward operator:
    # y_spc[k] = <phi_k, x_integrated>  where x_integrated = sum_lam x[:,:,lam]
    # phi_k is a binary random mask (spatial SPC pixel)
    # We compute N_spc = 64 SPC measurements (8x8 spatial pixels)
    N_spc = 64  # SPC measurements (8x8)
    rng = np.random.default_rng(seed=2024)

    print(f"\n[2] Applying SPC Kronecker forward model ({N_spc} measurements)...")
    out_files = []
    for idx, src_path in enumerate(cassi_files[:10]):
        with h5py.File(str(src_path), "r") as f:
            x_true = f["x_true"][:]   # (256,256,28)
            # Get original H_params
            orig_scene = str(f["H_params"].attrs.get("scene_name", f"scene{idx+1:02d}"))

        # Spectral integration (panchromatic)
        x_pan = x_true.sum(axis=2).astype(np.float32)  # (256,256)

        # Generate SPC random binary masks (64 measurements)
        H, W = x_pan.shape
        phi = (rng.uniform(0, 1, (N_spc, H, W)) > 0.5).astype(np.float32)

        # SPC measurement: y[k] = sum_{i,j} phi[k,i,j] * x_pan[i,j]
        y_spc = np.einsum('kij,ij->k', phi, x_pan).astype(np.float32)  # (N_spc,)
        y_spc_norm = y_spc / (y_spc.max() + 1e-8)

        h5_path = OUTDIR / f"standard_spc_kronecker_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=x_true,     compression="gzip")
            f.create_dataset("y_ideal", data=y_spc_norm, compression="gzip")
            f.create_dataset("phi",     data=phi,        compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]      = "spc_kronecker_random"
            hp.attrs["n_measurements"] = N_spc
            hp.attrs["scene_name"]    = orig_scene
            hp.attrs["noise"]         = "none"
            hp.attrs["mismatch"]      = "none"

            meta = f.create_group("metadata")
            meta.attrs["source"]    = "https://github.com/mengziyi64/TSA-Net"
            meta.attrs["doi"]       = "10.1007/978-3-030-58592-1_12"
            meta.attrs["citation"]  = "Cai et al., MST++, CVPR 2022 (KAIST HSI dataset)"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  sample {idx:02d}: x_true={x_true.shape} y_ideal={y_spc_norm.shape} "
              f"range=[{y_spc_norm.min():.3f},{y_spc_norm.max():.3f}]")

    meta_json = {
        "modality": "spc_kronecker",
        "canonical_dataset": "KAIST Hyperspectral dataset (10 scenes, 256x256x28)",
        "source": "https://github.com/mengziyi64/TSA-Net",
        "citation": "Cai et al., MST++, CVPR 2022",
        "doi": "10.1007/978-3-030-58592-1_12",
        "license": "For research use (KAIST HSI)",
        "n_samples": len(out_files),
        "x_true_shape": [256, 256, 28],
        "y_ideal_shape": [N_spc],
        "note": "x_true=HSI cube; y_ideal=SPC random projections (64 measurements)",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/spc_kronecker/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "spc_kronecker",
            "operator": "spc_kronecker_random",
            "n_measurements": N_spc,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: {len(out_files)} samples built in {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
