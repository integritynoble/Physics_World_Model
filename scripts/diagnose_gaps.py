#!/usr/bin/env python3
"""Diagnose why close-gap modalities have remaining PSNR gaps.

For each close-gap modality:
- Check x_true vs y relationship
- Check if forward model is too hard
- Check H_ideal properties
- Check reconstruction_baseline quality
- Suggest specific fixes
"""
import os, sys, glob, json
import numpy as np
import h5py
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(ROOT, "datasets", "benchmark")

# Close-gap modalities from analysis (3-5 dB gap entries)
CLOSE_GAP = [
    "sted", "spectral_ct", "spect", "ct", "light_field", "pump_probe",
    "cars", "passive_microwave", "radio_interferometry", "diffusion_mri",
    "machine_vision", "ultrasound", "event_camera", "cassi", "nsom",
    "nerf", "doppler_ultrasound", "two_photon", "us_mri", "photoacoustic",
    "polsar", "solar_imaging", "confocal_3d", "bioluminescence_tomo",
    "sem", "matrix", "dark_field",
    # Medium gap worth checking too
    "impedance_tomo", "tem", "holography", "mammography", "fib_sem",
    "three_photon", "cryo_em", "sim", "widefield", "ptychography",
    "ghost_imaging", "mri", "pet_ct", "fmri", "dot",
]


def psnr(gt, pred):
    gt, pred = gt.astype(np.float64), pred.astype(np.float64)
    mse = np.mean((gt - pred) ** 2)
    if mse < 1e-15: return 100.0
    mx = max(np.max(np.abs(gt)), 1e-10)
    return float(10 * np.log10(mx**2 / mse))


def diagnose_modality(mod_name):
    """Analyze H5 data for a modality."""
    mod_dir = os.path.join(BENCHMARK_DIR, mod_name)
    h5_files = sorted(glob.glob(os.path.join(mod_dir, "**/*.h5"), recursive=True))
    if not h5_files:
        return None

    info = {"modality": mod_name, "h5_files": len(h5_files)}

    with h5py.File(h5_files[0], "r") as f:
        sample_keys = sorted([k for k in f.keys() if k.startswith("sample_")])
        if not sample_keys:
            return None

        s = f[sample_keys[0]]
        info["keys"] = list(s.keys())

        if "x_true" not in s:
            return info

        x = s["x_true"][:]
        info["x_shape"] = list(x.shape)
        info["x_dtype"] = str(x.dtype)
        info["x_range"] = [float(np.min(x)), float(np.max(x))]
        info["x_mean"] = float(np.mean(x))
        info["x_std"] = float(np.std(x))

        if "y" in s:
            y = s["y"][:]
            info["y_shape"] = list(y.shape)
            info["y_dtype"] = str(y.dtype)
            info["y_range"] = [float(np.min(y)), float(np.max(y))]
            info["y_mean"] = float(np.mean(y))
            info["y_std"] = float(np.std(y))

            # Check if y is just noisy x
            if y.shape == x.shape:
                info["y_eq_x_psnr"] = round(psnr(x, y), 2)
                info["forward_type"] = "same_shape"
            elif y.size < x.size:
                info["compression"] = round(x.size / max(y.size, 1), 2)
                info["forward_type"] = "compressed"
            elif y.size > x.size:
                info["expansion"] = round(y.size / max(x.size, 1), 2)
                info["forward_type"] = "expanded"
        else:
            info["forward_type"] = "no_y"

        if "H_ideal" in s:
            H = s["H_ideal"][:]
            info["H_shape"] = list(H.shape)
            info["H_dtype"] = str(H.dtype)
            info["H_range"] = [float(np.min(H)), float(np.max(H))]
            info["H_nnz_ratio"] = round(float(np.count_nonzero(H)) / max(H.size, 1), 4)

        if "reconstruction_baseline" in s:
            bl = s["reconstruction_baseline"][:]
            info["bl_shape"] = list(bl.shape)
            info["bl_range"] = [float(np.min(bl)), float(np.max(bl))]
            if bl.shape == x.shape:
                info["baseline_psnr"] = round(psnr(x, bl), 2)
            else:
                info["baseline_psnr"] = 0.0
                info["shape_mismatch"] = True

            # Check if baseline is well-normalized
            if x.shape == bl.shape:
                x_max = max(np.max(x), 1e-10)
                bl_max = max(np.max(bl), 1e-10)
                info["range_ratio"] = round(bl_max / x_max, 3)

                # Test simple rescaling
                scale = x_max / bl_max
                bl_rescaled = np.clip(bl * scale, 0, x_max)
                rescaled_psnr = psnr(x, bl_rescaled)
                if rescaled_psnr > info["baseline_psnr"] + 0.5:
                    info["rescale_gain"] = round(rescaled_psnr - info["baseline_psnr"], 2)
                    info["rescaled_psnr"] = round(rescaled_psnr, 2)

        # Count all samples
        psnr_vals = []
        for sk in sample_keys[:10]:
            try:
                sx = f[sk]
                if "x_true" in sx and "reconstruction_baseline" in sx:
                    xx = sx["x_true"][:]
                    bb = sx["reconstruction_baseline"][:]
                    if xx.shape == bb.shape:
                        psnr_vals.append(psnr(xx, bb))
            except Exception:
                pass
        if psnr_vals:
            info["mean_psnr"] = round(float(np.mean(psnr_vals)), 2)
            info["n_samples"] = len(psnr_vals)

    return info


def main():
    results = {}
    for mod in CLOSE_GAP:
        mod_dir = os.path.join(BENCHMARK_DIR, mod)
        if not os.path.isdir(mod_dir):
            continue

        info = diagnose_modality(mod)
        if info is None:
            print(f"{mod:30s}: no data")
            continue

        results[mod] = info

        bl_psnr = info.get("baseline_psnr", 0)
        fwd_type = info.get("forward_type", "?")
        x_shape = info.get("x_shape", [])
        y_shape = info.get("y_shape", [])
        rescale_gain = info.get("rescale_gain", 0)

        flags = []
        if rescale_gain > 0.5:
            flags.append(f"RESCALE+{rescale_gain:.1f}dB")
        if fwd_type == "same_shape":
            eq_psnr = info.get("y_eq_x_psnr", 0)
            if eq_psnr > bl_psnr:
                flags.append(f"y_IS_BETTER({eq_psnr:.1f}dB)")
        rr = info.get("range_ratio", 1.0)
        if rr < 0.5 or rr > 2.0:
            flags.append(f"RANGE_MISMATCH(x/bl={rr:.2f})")

        flag_str = " ".join(flags) if flags else ""
        print(f"{mod:30s}: bl={bl_psnr:6.1f} dB | {fwd_type:12s} | x={x_shape} y={y_shape} | {flag_str}")

    # Save
    out_path = os.path.join(ROOT, "benchmark_results", "gap_diagnosis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved diagnosis to {out_path}")


if __name__ == "__main__":
    main()
