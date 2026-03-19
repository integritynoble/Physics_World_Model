#!/usr/bin/env python3
# batch_recon_pnpcassi.py

"""
Batch test all .mat datasets in a folder, reconstruct with TWO Phi masks:
  - dataset['mask3d_real']
  - dataset['mask3d_assumed']

For each (file, phi_type) saves:
  - recon .mat (img + curves + final PSNR/SSIM + runtime + metadata)
  - optional PNG grid previews

Also writes:
  - summary.csv
  - summary_metrics.json  (what you asked)

Key guarantees:
  1) SSIM is computed on final shift_back recon (if scikit-image installed),
     and saved to MAT + summary_metrics.json.
  2) Handles truth/mask layout mismatch (C,H,W vs H,W,C).
  3) Default CASSI shift step is 2 (GAP/ADMM).
  4) Robust to scikit-image API changes (n_iter_max/max_num_iter, multichannel/channel_axis).
"""

import os
import time
import glob
import json
import csv
import argparse
import inspect
from typing import Optional, Tuple, List, Any, Dict

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# (0) Patch scikit-image TV API mismatch BEFORE importing dvp_linear_inv_cassi
# ------------------------------------------------------------
def _patch_skimage_tv_chambolle() -> None:
    try:
        import skimage  # noqa: F401
        import skimage.restoration as _skr

        if not hasattr(_skr, "denoise_tv_chambolle"):
            print("[WARN] skimage.restoration.denoise_tv_chambolle not found; skip patch")
            return

        _orig = _skr.denoise_tv_chambolle
        _params = inspect.signature(_orig).parameters

        def _compat(*args, **kwargs):
            # map n_iter_max -> max_num_iter (newer skimage)
            if "n_iter_max" in kwargs and "max_num_iter" in _params and "max_num_iter" not in kwargs:
                kwargs["max_num_iter"] = kwargs.pop("n_iter_max")

            # map multichannel -> channel_axis (newer skimage)
            if "multichannel" in kwargs and "channel_axis" in _params and "channel_axis" not in kwargs:
                mc = bool(kwargs.pop("multichannel"))
                kwargs["channel_axis"] = (-1 if mc else None)

            # drop args that current version doesn't accept
            for k in list(kwargs.keys()):
                if k not in _params:
                    kwargs.pop(k, None)

            return _orig(*args, **kwargs)

        _skr.denoise_tv_chambolle = _compat
        print("[PATCH] denoise_tv_chambolle: n_iter_max->max_num_iter, multichannel->channel_axis")

    except Exception as e:
        print("[WARN] Could not patch skimage TV API:", repr(e))

_patch_skimage_tv_chambolle()

# ------------------------------------------------------------
# (1) Now safe to import your recon code (it may import skimage internally)
# ------------------------------------------------------------
from dvp_linear_inv_cassi import gap_denoise, admm_denoise
from utils import A, At, psnr as psnr_fn, shift_back

# Optional SSIM
try:
    from skimage.metrics import structural_similarity as ssim_fn  # type: ignore
    _HAS_SKIMAGE = True
except Exception:
    ssim_fn = None
    _HAS_SKIMAGE = False


# ----------------------------
# IO helpers
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def loadmat_any(path: str) -> Dict[str, Any]:
    """
    Load both old MAT and v7.3 MAT safely.
    """
    # try v7.3 via h5py first (only if needed)
    try:
        import h5py  # type: ignore
        with h5py.File(path, "r") as f:
            # If it opens, we still might prefer scipy for simplicity,
            # but many v7.3 files need h5py.
            # We'll convert datasets we need later by reading keys from f.
            data = {"__h5py__": f}
            data["__path__"] = path
            data["__keys__"] = list(f.keys())
            return data
    except Exception:
        pass

    # fallback to scipy
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)

def h5_get(f, key: str):
    # h5py datasets are often stored as column-major; we will squeeze and transpose if needed later.
    if key not in f:
        return None
    arr = f[key][()]
    return arr

def safe_get(data: Dict[str, Any], key: str):
    # if h5py-backed
    if "__h5py__" in data:
        f = data["__h5py__"]
        return h5_get(f, key)
    return data.get(key, None)

def pick_truth_key(data: Dict[str, Any]) -> Optional[str]:
    for k in ["orig", "truth", "gt"]:
        v = safe_get(data, k)
        if v is not None:
            return k
    return None


# ----------------------------
# Shape / normalization helpers
# ----------------------------
def to_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

def squeeze_meas(meas) -> np.ndarray:
    meas = np.asarray(meas)
    meas = np.squeeze(meas)
    if meas.ndim != 2:
        raise ValueError(f"meas must be 2D after squeeze, got {meas.shape}")
    return meas.astype(np.float32, copy=False)

def to_hwc_3d(arr: np.ndarray, name: str) -> np.ndarray:
    """
    Convert 3D array to HxWxC.
    Common cases:
      - HxWxC already
      - CxHxW (C small)
      - HxC xW (rare) -> not handled unless obvious
    """
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3D, got {arr.shape}")

    # If first dim is small -> likely C,H,W
    if arr.shape[0] <= 64 and arr.shape[1] > 64 and arr.shape[2] > 64:
        arr = np.transpose(arr, (1, 2, 0))

    return arr.astype(np.float32, copy=False)

def normalize_if_255(truth_hwc: np.ndarray, meas_hw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    If truth max suggests 0..255, divide both truth and meas by 255.
    Returns (truth_norm, meas_norm, scale_used).
    """
    mx = float(np.max(truth_hwc))
    scale = 255.0 if mx > 1.5 else 1.0
    if scale != 1.0:
        truth_hwc = truth_hwc / scale
        meas_hw = meas_hw / scale
    truth_hwc = np.clip(truth_hwc, 0.0, 1.0)
    # meas is a sum across bands; do NOT clip meas
    return truth_hwc, meas_hw, scale

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x.astype(np.float32, copy=False), 0.0, 1.0)

def align_and_crop(a_hwc: np.ndarray, b_hwc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int]]:
    """
    Crop both to common min (H,W,C) from top-left.
    Returns (a_crop, b_crop, (H,W,C)).
    """
    Ha, Wa, Ca = a_hwc.shape
    Hb, Wb, Cb = b_hwc.shape
    H = min(Ha, Hb)
    W = min(Wa, Wb)
    C = min(Ca, Cb)
    return a_hwc[:H, :W, :C], b_hwc[:H, :W, :C], (H, W, C)


# ----------------------------
# Metrics
# ----------------------------
def compute_psnr_safe(x: np.ndarray, y: np.ndarray) -> float:
    """
    PSNR in [0,1]. If utils.psnr fails, fallback.
    """
    try:
        v = float(psnr_fn(x, y))
        if np.isfinite(v):
            return v
    except Exception:
        pass

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mse = float(np.mean((x - y) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))

def compute_ssim_volume(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], np.ndarray]:
    """
    SSIM per-band + mean SSIM, for HxWxC in [0,1].
    """
    if (not _HAS_SKIMAGE) or (ssim_fn is None):
        return None, np.array([], dtype=np.float32)

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 3 or y.ndim != 3:
        return None, np.array([], dtype=np.float32)

    if x.shape != y.shape:
        return None, np.array([], dtype=np.float32)

    C = x.shape[2]
    per = np.empty((C,), dtype=np.float32)
    for c in range(C):
        try:
            per[c] = float(ssim_fn(y[:, :, c], x[:, :, c], data_range=1.0))
        except Exception:
            per[c] = np.nan

    if np.all(~np.isfinite(per)):
        return None, per

    return float(np.nanmean(per)), per


# ----------------------------
# Visualization
# ----------------------------
def save_grid_png(vol_hwc: np.ndarray, out_png: str, max_bands: int = 25, grid_hw=(5, 5)) -> None:
    H, W, C = vol_hwc.shape
    n = min(max_bands, C, grid_hw[0] * grid_hw[1])
    fig = plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(grid_hw[0], grid_hw[1], i + 1)
        plt.imshow(vol_hwc[:, :, i], cmap=plt.cm.gray, vmin=0, vmax=1)
        plt.axis("off")
    plt.tight_layout(pad=0.1)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------------------
# Recon core
# ----------------------------
def parse_sigma(sigma_str: str, C: int) -> List[float]:
    sigma_str = (sigma_str or "").strip()
    if sigma_str == "":
        return [130.0] * C
    if "," not in sigma_str:
        return [float(sigma_str)] * C
    vals = [float(x.strip()) for x in sigma_str.split(",") if x.strip() != ""]
    if len(vals) == 0:
        return [130.0] * C
    if len(vals) < C:
        vals += [vals[-1]] * (C - len(vals))
    if len(vals) > C:
        vals = vals[:C]
    return vals

def to_np_array(x: Any, dtype=np.float32) -> np.ndarray:
    if x is None:
        return np.array([], dtype=dtype)
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    try:
        return np.asarray(x, dtype=dtype)
    except Exception:
        return np.array([], dtype=dtype)

def run_recon_one(meas_hw: np.ndarray, truth_hwc: Optional[np.ndarray], phi_hwc: np.ndarray, args):
    """
    meas_hw: (H, Wm) float32
    truth_hwc: (H, W, C) float32 in [0,1] or None
    phi_hwc: (H, Wm or W, C) float32
    """
    C = int(phi_hwc.shape[2])
    sigma_list = parse_sigma(args.sigma, C)

    t0 = time.time()

    method = args.method.upper()
    psnr_curve = None
    ssim_curve = None

    if method == "GAP":
        out = gap_denoise(
            meas_hw, phi_hwc, A, At,
            args.lam,
            args.accelerate,
            args.denoiser,
            args.iter_max,
            tv_weight=args.tv_weight,
            tv_iter_max=args.tv_iter_max,
            X_orig=truth_hwc,
            sigma=sigma_list
        )
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            vrec = out[0]
            psnr_curve = out[1]
            if len(out) >= 3:
                ssim_curve = out[2]
        else:
            vrec = out

        vrecon_hwc = shift_back(vrec, step=args.shift_step_gap)

    elif method == "ADMM":
        vrec, psnr_curve, ssim_curve = admm_denoise(
            meas_hw, phi_hwc, A, At,
            args.lam,
            args.gamma,
            args.denoiser.lower(),
            args.iter_max,
            tv_weight=args.tv_weight,
            tv_iter_max=args.tv_iter_max,
            X_orig=truth_hwc,
            sigma=sigma_list
        )
        vrecon_hwc = shift_back(vrec, step=args.shift_step_admm)

    else:
        raise ValueError(f"Unknown method {args.method}")

    runtime = float(time.time() - t0)

    # Ensure recon is float32 and clipped for metrics
    vrecon_hwc = to_hwc_3d(vrecon_hwc, "vrecon")
    vrecon_hwc = clip01(vrecon_hwc)

    psnr_curve_arr = to_np_array(psnr_curve, np.float32)
    ssim_curve_arr = to_np_array(ssim_curve, np.float32)

    psnr_last_curve = float(psnr_curve_arr[-1]) if psnr_curve_arr.size > 0 else np.nan
    ssim_last_curve = float(ssim_curve_arr[-1]) if ssim_curve_arr.size > 0 else np.nan

    # Final metrics on shiftback recon
    psnr_shiftback = np.nan
    ssim_shiftback = np.nan
    ssim_band_shiftback = np.array([], dtype=np.float32)
    crop_used = None

    if truth_hwc is not None:
        truth_hwc = clip01(truth_hwc)

        # align shapes if needed
        if vrecon_hwc.shape != truth_hwc.shape:
            vrecon_al, truth_al, shape_used = align_and_crop(vrecon_hwc, truth_hwc)
            crop_used = {"shape_used": shape_used, "recon_shape": vrecon_hwc.shape, "truth_shape": truth_hwc.shape}
        else:
            vrecon_al, truth_al = vrecon_hwc, truth_hwc

        psnr_shiftback = float(compute_psnr_safe(vrecon_al, truth_al))

        mean_ssim, per_band = compute_ssim_volume(vrecon_al, truth_al)
        if mean_ssim is not None:
            ssim_shiftback = float(mean_ssim)
            ssim_band_shiftback = per_band.astype(np.float32, copy=False)
        else:
            ssim_shiftback = np.nan
            ssim_band_shiftback = per_band.astype(np.float32, copy=False)

    info = {
        "method": method,
        "denoiser": args.denoiser,
        "runtime_sec": runtime,
        "psnr_curve": psnr_curve_arr,
        "ssim_curve": ssim_curve_arr,
        "psnr_last_curve": psnr_last_curve,
        "ssim_last_curve": ssim_last_curve,
        "psnr_shiftback": psnr_shiftback,
        "ssim_shiftback": ssim_shiftback,
        "ssim_band_shiftback": ssim_band_shiftback,
        "sigma_list": np.asarray(sigma_list, dtype=np.float32),
        "has_skimage": bool(_HAS_SKIMAGE),
        "crop_used": crop_used,
    }
    return vrecon_hwc, info


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_dir", type=str, default="/home/spiritai/InverseNet/SCI_spectral/simulate/datasetn")
    ap.add_argument("--out_dir", type=str, default="/home/spiritai/InverseNet/SCI_spectral/PnP-CASSI/result2")

    ap.add_argument("--method", type=str, default="GAP", choices=["GAP", "ADMM"])
    ap.add_argument("--denoiser", type=str, default="TV")

    ap.add_argument("--iter_max", type=int, default=20)
    ap.add_argument("--tv_weight", type=float, default=6.0)
    ap.add_argument("--tv_iter_max", type=int, default=5)

    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.02)

    ap.add_argument("--no_accelerate", action="store_true")
    ap.add_argument("--sigma", type=str, default="130")

    ap.add_argument("--no_png", action="store_true")
    ap.add_argument("--max_preview_bands", type=int, default=25)

    # IMPORTANT: CASSI default shift step is usually 2
    ap.add_argument("--shift_step_gap", type=int, default=2)
    ap.add_argument("--shift_step_admm", type=int, default=2)

    args = ap.parse_args()
    args.accelerate = not args.no_accelerate
    args.save_png = not args.no_png

    ensure_dir(args.out_dir)

    mat_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {args.dataset_dir}")

    summary_csv = os.path.join(args.out_dir, "summary.csv")
    summary_metrics_json = os.path.join(args.out_dir, "summary_metrics.json")

    rows: List[Dict[str, Any]] = []

    for fpath in mat_files:
        base = os.path.splitext(os.path.basename(fpath))[0]
        print(f"\n==== Processing: {base} ====")

        try:
            data = loadmat_any(fpath)
        except Exception as e:
            print(f"  [FAIL] loadmat: {repr(e)}")
            rows.append({"file": base, "status": "fail", "err": f"loadmat: {repr(e)}"})
            continue

        # meas
        try:
            meas_raw = safe_get(data, "meas")
            if meas_raw is None:
                raise KeyError("missing key 'meas'")
            meas_hw = squeeze_meas(meas_raw)
        except Exception as e:
            print(f"  [SKIP] meas error: {repr(e)}")
            rows.append({"file": base, "status": "skip", "err": f"meas: {repr(e)}"})
            continue

        # truth (optional)
        truth_hwc = None
        scale_used = 1.0
        truth_key = pick_truth_key(data)
        if truth_key is not None:
            try:
                truth_raw = safe_get(data, truth_key)
                truth_hwc = to_hwc_3d(truth_raw, truth_key)
                truth_hwc, meas_hw, scale_used = normalize_if_255(truth_hwc, meas_hw)
            except Exception as e:
                print(f"  [WARN] truth load error ({truth_key}): {repr(e)} -> metrics may be NaN")
                truth_hwc = None
        else:
            print("  [WARN] no truth key found (orig/truth/gt). metrics will be NaN")

        for phi_key in ["mask3d_real", "mask3d_assumed"]:
            tag = phi_key.replace("mask3d_", "")
            phi_raw = safe_get(data, phi_key)
            if phi_raw is None:
                print(f"  [SKIP] missing key '{phi_key}'")
                rows.append({
                    "file": base, "phi": tag, "status": "skip",
                    "err": f"missing {phi_key}"
                })
                continue

            try:
                phi_hwc = to_hwc_3d(phi_raw, phi_key)
            except Exception as e:
                print(f"  [FAIL] {phi_key} shape error: {repr(e)}")
                rows.append({"file": base, "phi": tag, "status": "fail", "err": f"{phi_key}: {repr(e)}"})
                continue

            # If we scaled by 255 using truth, we should scale phi? No. Phi is a mask (0/1), keep as is.

            try:
                vrecon_hwc, info = run_recon_one(meas_hw, truth_hwc, phi_hwc, args)
            except Exception as e:
                print(f"  [FAIL] {base} | {phi_key}: {repr(e)}")
                rows.append({
                    "file": base, "phi": tag, "status": "fail",
                    "psnr_shiftback": None, "ssim_shiftback": None,
                    "runtime_sec": None,
                    "err": repr(e),
                })
                continue

            out_mat = os.path.join(args.out_dir, f"{base}_{tag}_{info['method']}_{info['denoiser']}_recon.mat")

            mat_payload = {
                "img": vrecon_hwc.astype(np.float32, copy=False),

                "psnr_curve": to_np_array(info["psnr_curve"], np.float32),
                "ssim_curve": to_np_array(info["ssim_curve"], np.float32),

                "psnr_last_curve": np.array([info["psnr_last_curve"]], dtype=np.float32),
                "ssim_last_curve": np.array([info["ssim_last_curve"]], dtype=np.float32),

                # final metrics (ALWAYS scalar arrays, NaN if unavailable)
                "psnr_shiftback": np.array([info["psnr_shiftback"]], dtype=np.float32),
                "ssim_shiftback": np.array([info["ssim_shiftback"]], dtype=np.float32),
                "ssim_band_shiftback": to_np_array(info["ssim_band_shiftback"], np.float32),

                "runtime_sec": np.array([info["runtime_sec"]], dtype=np.float32),

                "method": info["method"],
                "denoiser": info["denoiser"],
                "phi_type": tag,
                "src_file": base,

                "scale_used": np.array([scale_used], dtype=np.float32),
                "shift_step_gap": np.array([args.shift_step_gap], dtype=np.int32),
                "shift_step_admm": np.array([args.shift_step_admm], dtype=np.int32),
                "has_skimage": np.array([int(info["has_skimage"])], dtype=np.int32),

                "sigma_list": to_np_array(info["sigma_list"], np.float32),
            }

            if info.get("crop_used") is not None:
                mat_payload["crop_used_json"] = json.dumps(info["crop_used"])

            if truth_hwc is not None:
                mat_payload["truth"] = truth_hwc.astype(np.float32, copy=False)

            sio.savemat(out_mat, mat_payload, do_compression=True)

            if args.save_png:
                try:
                    save_grid_png(vrecon_hwc, os.path.join(args.out_dir, f"{base}_{tag}_recon.png"),
                                  max_bands=args.max_preview_bands)
                    if truth_hwc is not None:
                        save_grid_png(truth_hwc, os.path.join(args.out_dir, f"{base}_{tag}_truth.png"),
                                      max_bands=args.max_preview_bands)
                except Exception as e:
                    print(f"  [WARN] PNG save failed for {base} {tag}: {repr(e)}")

            print(
                f"  [{tag}] PSNR(sb)={info['psnr_shiftback']:.4f} "
                f"SSIM(sb)={info['ssim_shiftback']:.4f} "
                f"time={info['runtime_sec']:.1f}s -> {os.path.basename(out_mat)}"
            )

            rows.append({
                "file": base,
                "phi": tag,
                "status": "ok",
                "psnr_shiftback": None if np.isnan(info["psnr_shiftback"]) else float(info["psnr_shiftback"]),
                "ssim_shiftback": None if np.isnan(info["ssim_shiftback"]) else float(info["ssim_shiftback"]),
                "psnr_last_curve": None if np.isnan(info["psnr_last_curve"]) else float(info["psnr_last_curve"]),
                "ssim_last_curve": None if np.isnan(info["ssim_last_curve"]) else float(info["ssim_last_curve"]),
                "runtime_sec": float(info["runtime_sec"]),
                "out_mat": os.path.basename(out_mat),
                "method": info["method"],
                "denoiser": info["denoiser"],
                "scale_used": float(scale_used),
                "has_skimage": bool(info["has_skimage"]),
                "crop_used": info.get("crop_used"),
                "err": "",
            })

    # Write CSV
    fieldnames = [
        "file", "phi", "status",
        "psnr_shiftback", "ssim_shiftback",
        "psnr_last_curve", "ssim_last_curve",
        "runtime_sec", "out_mat",
        "method", "denoiser",
        "scale_used", "has_skimage",
        "err",
    ]
    with open(summary_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # ensure all keys exist
            for k in fieldnames:
                if k not in r:
                    r[k] = ""
            w.writerow(r)

    # Write JSON summary (metrics-focused)
    with open(summary_metrics_json, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2)

    print(f"\nDone.\n- CSV:  {summary_csv}\n- JSON: {summary_metrics_json}\n- MATs in: {args.out_dir}")
    if not _HAS_SKIMAGE:
        print("Note: scikit-image not found -> SSIM will be NaN/None. Install: pip install scikit-image")


if __name__ == "__main__":
    main()
