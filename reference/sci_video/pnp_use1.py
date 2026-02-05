#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust batch runner for PnP-SCI (SCI-video) that:

1) Runs EVERY .mat in a folder (does NOT stop if one file fails)
2) For each dataset, runs multiple denoisers (TV / FFDNet / FastDVDnet)
3) Saves per-dataset .mat results (PSNR/SSIM arrays + runtime + optional recon)
4) Writes a summary CSV for all datasets & all methods
5) Supports MATLAB v7.3 via h5py, and v7.2- via scipy.io.loadmat

Notes:
- This script keeps the same core call: admmdenoise_cacti(...)
- It catches exceptions per (dataset, method) so the batch always continues.
- It preloads FFDNet / FastDVDnet ONCE outside the dataset loop (faster, more stable).

If SSIM import issues exist, ensure you have scikit-image in this env:
  conda install -c conda-forge scikit-image
"""

import os
import time
import math
import glob
import csv
from statistics import mean

import numpy as np
import scipy.io as sio

# v7.3 support
import h5py

from pnp_sci import admmdenoise_cacti
from utils import (A_, At_)

# ----------------------------
# Config
# ----------------------------
datasetdir = "/home/spiritai/InverseNet/SCI_video/simulate/test/dataset/sci_video_benchmark_v8/dataset"
resultsdir = "/home/spiritai/PnP-SCI_python-master/results/use"

# If you want to limit number of frames for speed, set e.g. 1 or 8
# If -1, will use all frames in the file.
DEFAULT_NFRAME = -1

# Save reconstructed videos? Can be huge. If you only need PSNR/SSIM, set False.
SAVE_RECON = True

# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_mat_files(folder: str):
    if os.path.isdir(folder):
        files = sorted(glob.glob(os.path.join(folder, "*.mat")))
        return files
    if folder.endswith(".mat"):
        return [folder]
    raise ValueError(f"datasetdir must be a folder or a .mat file. Got: {folder}")

def load_mat_any(mat_path: str):
    """
    Return (meas, mask, orig) as float32 with correct orientation.
    Works for both v7.3 (h5py) and classic .mat (scipy.io.loadmat).
    Expected keys: meas, mask, orig
    """
    # Try v7.3 first
    try:
        with h5py.File(mat_path, "r") as f:
            # h5py loads in column-major-like; your original code transposed.
            meas = np.float32(f["meas"][:]).transpose()
            mask = np.float32(f["mask"][:]).transpose()
            orig = np.float32(f["orig"][:]).transpose()
            return meas, mask, orig
    except Exception:
        pass

    # Fall back to classic mat
    d = sio.loadmat(mat_path)
    if "meas" not in d or "mask" not in d or "orig" not in d:
        raise KeyError(f"Missing one of keys meas/mask/orig in {mat_path}. Keys={list(d.keys())}")
    meas = np.float32(d["meas"])
    mask = np.float32(d["mask"])
    orig = np.float32(d["orig"])
    return meas, mask, orig

def safe_mean(x):
    try:
        return float(mean(x))
    except Exception:
        # x might be numpy scalar/array
        x = np.asarray(x).astype(np.float32).ravel()
        if x.size == 0:
            return float("nan")
        return float(np.nanmean(x))

def np_array(x):
    return np.asarray(x)

# ----------------------------
# Optional: preload denoiser networks
# ----------------------------
_USE_GPU = True

def preload_ffdnet(model_fn: str, in_ch: int = 1):
    import torch
    from packages.ffdnet.models import FFDNet

    net = FFDNet(num_input_channels=in_ch)
    state_dict = torch.load(model_fn, map_location="cuda" if _USE_GPU else "cpu")

    if _USE_GPU:
        device_ids = [0]
        model = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
    else:
        model = net

    model.load_state_dict(state_dict)
    model.eval()
    return model

def preload_fastdvdnet(model_fn: str, num_input_frames: int = 5, num_color_channels: int = 1):
    import torch
    from packages.fastdvdnet.models import FastDVDnet

    model = FastDVDnet(num_input_frames=num_input_frames, num_color_channels=num_color_channels)
    state_dict = torch.load(model_fn, map_location="cuda" if _USE_GPU else "cpu")

    if _USE_GPU:
        model = model.cuda()

    model.load_state_dict(state_dict)
    model.eval()
    return model

# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dir(resultsdir)

    mat_files = list_mat_files(datasetdir)
    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found under: {datasetdir}")

    # Output folders
    savedmatdir = os.path.join(resultsdir, "savedmat", "cacti_batch")
    ensure_dir(savedmatdir)

    # Summary CSV
    summary_csv = os.path.join(resultsdir, "summary_batch.csv")
    with open(summary_csv, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "file",
            "method",
            "denoiser",
            "nmask",
            "nframe",
            "psnr_mean",
            "ssim_mean",
            "runtime_sec",
            "status",
            "err",
            "out_mat",
        ])
        writer.writeheader()

        # Preload models once
        ffdnet_model = None
        fastdvdnet_model = None

        try:
            # You can comment out a model if you don't want to run it
            ffdnet_model_path = "/home/spiritai/PnP-SCI_python-master/packages/ffdnet/models/net_gray.pth"
            ffdnet_model = preload_ffdnet(ffdnet_model_path, in_ch=1)
            print("[OK] Loaded FFDNet model.")
        except Exception as e:
            print("[WARN] Failed to load FFDNet model:", repr(e))
            ffdnet_model = None

        try:
            fastdvdnet_model_path = "/home/spiritai/PnP-SCI_python-master/packages/fastdvdnet/model_gray.pth"
            fastdvdnet_model = preload_fastdvdnet(fastdvdnet_model_path, num_input_frames=5, num_color_channels=1)
            print("[OK] Loaded FastDVDnet model.")
        except Exception as e:
            print("[WARN] Failed to load FastDVDnet model:", repr(e))
            fastdvdnet_model = None

        # Loop all datasets
        for idx, matfile in enumerate(mat_files):
            datname = os.path.basename(matfile)
            base = os.path.splitext(datname)[0]
            print(f"\n==== [{idx+1}/{len(mat_files)}] Processing: {datname} ====")

            # Load data (dataset-level try: if load fails, skip to next file)
            try:
                meas, mask, orig = load_mat_any(matfile)
                print("  shapes:", meas.shape, mask.shape, orig.shape)
            except Exception as e:
                print("  [FAIL] load:", repr(e))
                writer.writerow({
                    "file": datname,
                    "method": "",
                    "denoiser": "",
                    "nmask": "",
                    "nframe": "",
                    "psnr_mean": "",
                    "ssim_mean": "",
                    "runtime_sec": "",
                    "status": "fail_load",
                    "err": repr(e),
                    "out_mat": "",
                })
                continue

            # Determine frames
            iframe = 0
            nframe = DEFAULT_NFRAME
            if nframe < 0:
                # meas expected H x W x T (or H x W x nframe). Your original used meas.shape[2]
                if meas.ndim < 3:
                    print("  [FAIL] meas has no time dimension:", meas.shape)
                    writer.writerow({
                        "file": datname,
                        "method": "",
                        "denoiser": "",
                        "nmask": "",
                        "nframe": "",
                        "psnr_mean": "",
                        "ssim_mean": "",
                        "runtime_sec": "",
                        "status": "fail_meas_shape",
                        "err": f"meas.ndim={meas.ndim}, expected >=3",
                        "out_mat": "",
                    })
                    continue
                nframe = meas.shape[2]

            MAXB = 255.0
            nmask = int(mask.shape[2]) if (mask is not None and mask.ndim == 3) else -1

            # Forward model
            A  = lambda x: A_(x, mask)
            At = lambda y: At_(y, mask)

            # Methods to run (each entry is one run)
            runs = []

            # --- GAP-TV ---
            runs.append(dict(
                projmeth="gap",
                denoiser="tv",
                _lambda=1,
                accelerate=True,
                iter_max=40,
                tv_weight=0.3,
                tv_iter_max=5,
                model=None,
                sigma=None,
                name="GAP-TV",
            ))

            # --- GAP-FFDNet ---
            if ffdnet_model is not None:
                runs.append(dict(
                    projmeth="gap",
                    denoiser="ffdnet",
                    _lambda=1,
                    accelerate=True,
                    iter_max=[10, 10, 10, 10],
                    tv_weight=None,
                    tv_iter_max=None,
                    model=ffdnet_model,
                    sigma=[50/255, 25/255, 12/255, 6/255],
                    name="GAP-FFDNet",
                ))
            else:
                print("  [SKIP] FFDNet run (model not loaded).")

            # --- GAP-FastDVDnet ---
            if fastdvdnet_model is not None:
                runs.append(dict(
                    projmeth="gap",
                    denoiser="fastdvdnet",
                    _lambda=1,
                    accelerate=True,
                    iter_max=[20, 20, 20, 20],
                    tv_weight=None,
                    tv_iter_max=None,
                    model=fastdvdnet_model,
                    sigma=[100/255, 50/255, 25/255, 12/255],
                    name="GAP-FastDVDnet",
                ))
            else:
                print("  [SKIP] FastDVDnet run (model not loaded).")

            # Execute runs (per-run try/except so batch never stops)
            for run in runs:
                projmeth = run["projmeth"]
                denoiser = run["denoiser"]
                run_name = run["name"]
                out_mat = os.path.join(savedmatdir, f"{base}_{run_name}_nmask{nmask}_nframe{nframe}.mat")

                print(f"  -- Running {run_name} ...")

                try:
                    t0 = time.time()
                    # Call admmdenoise_cacti with only relevant params
                    kwargs = dict(
                        projmeth=projmeth,
                        v0=None,
                        orig=orig,
                        iframe=iframe,
                        nframe=nframe,
                        MAXB=MAXB,
                        maskdirection="plain",
                        _lambda=run["_lambda"],
                        accelerate=run["accelerate"],
                        denoiser=denoiser,
                        iter_max=run["iter_max"],
                    )

                    if denoiser == "tv":
                        kwargs["tv_weight"] = run["tv_weight"]
                        kwargs["tv_iter_max"] = run["tv_iter_max"]

                    if run["model"] is not None:
                        kwargs["model"] = run["model"]
                    if run["sigma"] is not None:
                        kwargs["sigma"] = run["sigma"]

                    # returns: v, t, psnr, ssim, psnrall
                    v, t_run, psnr_arr, ssim_arr, psnrall = admmdenoise_cacti(meas, mask, A, At, **kwargs)

                    runtime_sec = float(t_run) if t_run is not None else float(time.time() - t0)
                    psnr_mean = safe_mean(psnr_arr)
                    ssim_mean = safe_mean(ssim_arr)

                    print(f"     {run_name}: PSNR {psnr_mean:.2f} dB, SSIM {ssim_mean:.4f}, time {runtime_sec:.1f}s")

                    # Save per-dataset, per-method
                    payload = {
                        "file": datname,
                        "base": base,
                        "projmeth": projmeth,
                        "denoiser": denoiser,
                        "nmask": np.array([nmask], dtype=np.int32),
                        "iframe": np.array([iframe], dtype=np.int32),
                        "nframe": np.array([nframe], dtype=np.int32),
                        "runtime_sec": np.array([runtime_sec], dtype=np.float32),
                        "psnr": np_array(psnr_arr).astype(np.float32),
                        "ssim": np_array(ssim_arr).astype(np.float32),
                        "psnrall": np_array(psnrall).astype(np.float32) if psnrall is not None else np.array([], dtype=np.float32),
                    }

                    if SAVE_RECON:
                        payload["recon"] = v  # can be huge

                    sio.savemat(out_mat, payload, do_compression=True)

                    writer.writerow({
                        "file": datname,
                        "method": projmeth.upper(),
                        "denoiser": denoiser.upper(),
                        "nmask": nmask,
                        "nframe": nframe,
                        "psnr_mean": f"{psnr_mean:.6f}",
                        "ssim_mean": f"{ssim_mean:.6f}",
                        "runtime_sec": f"{runtime_sec:.3f}",
                        "status": "ok",
                        "err": "",
                        "out_mat": os.path.basename(out_mat),
                    })

                except Exception as e:
                    print(f"     [FAIL] {run_name}: {repr(e)}")
                    writer.writerow({
                        "file": datname,
                        "method": projmeth.upper(),
                        "denoiser": denoiser.upper(),
                        "nmask": nmask,
                        "nframe": nframe,
                        "psnr_mean": "",
                        "ssim_mean": "",
                        "runtime_sec": "",
                        "status": "fail_run",
                        "err": repr(e),
                        "out_mat": "",
                    })
                    # continue to next run

    print("\nDone.")
    print(" - Per-dataset MATs:", savedmatdir)
    print(" - Summary CSV:", summary_csv)


if __name__ == "__main__":
    main()
