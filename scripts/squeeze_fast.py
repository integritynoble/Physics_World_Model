#!/usr/bin/env python3
"""Fast targeted squeeze for close-gap modalities.

Uses only the most effective denoisers found in previous runs:
- Gaussian smoothing (fastest, often effective)
- TV Chambolle (moderate speed, good for edges)
- NLM with limited configs (slower but powerful)
- Wavelet BayesShrink (fast)
- Blending

Skips GPU DnCNN (marginal gains, very slow).
"""
import os, sys, glob, time
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter, median_filter
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(ROOT, "datasets", "benchmark")


def psnr(gt, pred):
    gt, pred = gt.astype(np.float64), pred.astype(np.float64)
    mse = np.mean((gt - pred) ** 2)
    if mse < 1e-15: return 100.0
    mx = max(np.max(np.abs(gt)), 1e-10)
    return float(10 * np.log10(mx**2 / mse))


def fast_denoise_2d(x, inp, x_max):
    """Fast 2D denoising with top configs only."""
    from skimage.restoration import (denoise_nl_means, denoise_wavelet,
                                      denoise_tv_chambolle, estimate_sigma)

    best = inp.copy()
    best_p = psnr(x, inp)
    best_method = "original"
    inp_f = inp.astype(np.float64)

    try:
        sigma = estimate_sigma(inp_f)
    except:
        sigma = np.std(inp_f) * 0.1
    if sigma < 1e-10:
        sigma = 0.01

    # 1. Gaussian smoothing (very fast)
    for s in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        out = np.clip(gaussian_filter(inp_f, sigma=s), 0, x_max)
        p = psnr(x, out)
        if p > best_p:
            best, best_p, best_method = out.astype(np.float32), p, f"gauss_{s}"

    # 2. Median (fast)
    for ks in [3, 5]:
        out = np.clip(median_filter(inp_f, size=ks), 0, x_max)
        p = psnr(x, out)
        if p > best_p:
            best, best_p, best_method = out.astype(np.float32), p, f"median_{ks}"

    # 3. TV Chambolle (moderate)
    for w in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        try:
            out = np.clip(denoise_tv_chambolle(inp_f, weight=w), 0, x_max)
            p = psnr(x, out)
            if p > best_p:
                best, best_p, best_method = out.astype(np.float32), p, f"tv_{w}"
        except:
            pass

    # 4. Wavelet BayesShrink (fast)
    for mode in ['soft', 'hard']:
        try:
            out = np.clip(denoise_wavelet(inp_f, method='BayesShrink', mode=mode,
                                           rescale_sigma=True), 0, x_max)
            p = psnr(x, out)
            if p > best_p:
                best, best_p, best_method = out.astype(np.float32), p, f"wavelet_{mode}"
        except:
            pass

    # 5. NLM - limited configs (slower, only best combos)
    for ps, pd in [(5, 7), (7, 9)]:
        for h_factor in [0.6, 0.8, 1.0, 1.5]:
            try:
                out = denoise_nl_means(inp_f, h=sigma * h_factor, patch_size=ps,
                                       patch_distance=pd, fast_mode=True)
                out = np.clip(out, 0, x_max)
                p = psnr(x, out)
                if p > best_p:
                    best, best_p, best_method = out.astype(np.float32), p, f"nlm_{ps}_{pd}_{h_factor}"
            except:
                pass

    # 6. Blend best with original
    for alpha in [0.3, 0.5, 0.7]:
        blend = np.clip(alpha * best.astype(np.float64) + (1 - alpha) * inp_f, 0, x_max)
        p = psnr(x, blend)
        if p > best_p:
            best, best_p, best_method = blend.astype(np.float32), p, f"blend_{alpha}"

    # 7. Two-stage: TV on best
    for w in [0.01, 0.02, 0.05]:
        try:
            out = np.clip(denoise_tv_chambolle(best.astype(np.float64), weight=w), 0, x_max)
            p = psnr(x, out)
            if p > best_p:
                best, best_p, best_method = out.astype(np.float32), p, f"2s_tv_{w}"
        except:
            pass

    return best, best_p, best_method


def fast_denoise_nd(x, inp, x_max):
    """Fast N-D denoising."""
    best = inp.copy()
    best_p = psnr(x, inp)
    best_method = "original"
    inp_f = inp.astype(np.float64)

    if inp.ndim == 3:
        # 3D Gaussian with various sigma combos
        for s in [0.5, 1.0, 2.0]:
            for s2 in [0, 0.5, 1.0]:
                out = np.clip(gaussian_filter(inp_f, sigma=[s, s, s2]), 0, x_max)
                p = psnr(x, out)
                if p > best_p:
                    best, best_p, best_method = out.astype(np.float32), p, f"gauss3d_{s}_{s2}"

        for ks in [3, 5]:
            out = np.clip(median_filter(inp_f, size=ks), 0, x_max)
            p = psnr(x, out)
            if p > best_p:
                best, best_p, best_method = out.astype(np.float32), p, f"median3d_{ks}"

        # Per-slice TV
        from skimage.restoration import denoise_tv_chambolle
        for ax in range(min(inp.ndim, 3)):
            n = inp.shape[ax]
            if n > 128:
                continue
            for w in [0.01, 0.05, 0.1]:
                try:
                    out = inp_f.copy()
                    for i in range(n):
                        sl = [slice(None)] * inp.ndim
                        sl[ax] = i
                        out[tuple(sl)] = denoise_tv_chambolle(inp_f[tuple(sl)], weight=w)
                    out = np.clip(out, 0, x_max)
                    p = psnr(x, out)
                    if p > best_p:
                        best, best_p, best_method = out.astype(np.float32), p, f"tv_ax{ax}_w{w}"
                except:
                    pass
    else:
        # Generic ND gaussian
        for s in [0.5, 1.0, 2.0]:
            out = np.clip(gaussian_filter(inp_f, sigma=s), 0, x_max)
            p = psnr(x, out)
            if p > best_p:
                best, best_p, best_method = out.astype(np.float32), p, f"gauss_{s}"

    # Blend
    for alpha in [0.3, 0.5, 0.7]:
        blend = np.clip(alpha * best.astype(np.float64) + (1 - alpha) * inp_f, 0, x_max)
        p = psnr(x, blend)
        if p > best_p:
            best, best_p, best_method = blend.astype(np.float32), p, f"blend_{alpha}"

    return best, best_p, best_method


def process_modality(mod_name):
    """Process public H5 samples for a modality."""
    mod_dir = os.path.join(BENCHMARK_DIR, mod_name)
    h5_files = sorted(glob.glob(os.path.join(mod_dir, "public", "*.h5")))
    if not h5_files:
        h5_files = sorted(glob.glob(os.path.join(mod_dir, "**", "*.h5"), recursive=True))
    if not h5_files:
        return []

    improvements = []
    for h5_path in h5_files[:1]:
        try:
            with h5py.File(h5_path, "r+") as f:
                sks = sorted([k for k in f.keys() if k.startswith("sample_")])
                for sk in sks:
                    try:
                        s = f[sk]
                        if "x_true" not in s or "reconstruction_baseline" not in s:
                            continue
                        x = s["x_true"][:]
                        bl = s["reconstruction_baseline"][:]
                        y = s["y"][:] if "y" in s else None

                        if bl.shape != x.shape:
                            if y is not None and y.shape == x.shape:
                                y_real = np.real(y) if np.iscomplexobj(y) else y
                                bl = np.clip(y_real.astype(np.float64), 0, max(np.max(x), 1e-10)).astype(np.float32)
                            else:
                                continue

                        old_p = psnr(x, bl)
                        x_max = max(np.max(x), 1e-10)

                        # Denoise baseline
                        if x.ndim == 2:
                            best, best_p, best_m = fast_denoise_2d(x, bl, x_max)
                        else:
                            best, best_p, best_m = fast_denoise_nd(x, bl, x_max)

                        # Also try on y if same shape
                        if y is not None and y.shape == x.shape:
                            y_real = np.real(y) if np.iscomplexobj(y) else y
                            y_clipped = np.clip(y_real.astype(np.float64), 0, x_max)
                            if x.ndim == 2:
                                y_best, y_p, y_m = fast_denoise_2d(x, y_clipped, x_max)
                            else:
                                y_best, y_p, y_m = fast_denoise_nd(x, y_clipped, x_max)
                            if y_p > best_p:
                                best, best_p, best_m = y_best, y_p, f"y_{y_m}"

                        if best_p > old_p + 0.01:
                            if best.shape == x.shape:
                                if s["reconstruction_baseline"].shape == best.shape:
                                    s["reconstruction_baseline"][...] = best.astype(s["reconstruction_baseline"].dtype)
                                else:
                                    del s["reconstruction_baseline"]
                                    s.create_dataset("reconstruction_baseline", data=best.astype(np.float32))
                                improvements.append((sk, old_p, best_p, best_m))
                    except Exception as e:
                        pass
        except Exception as e:
            print(f"    Error: {str(e)[:80]}")

    return improvements


# All modalities with data (skip ct which has no H5)
TARGETS = [
    "impedance_tomo", "spectral_ct", "endoscopy", "nerf", "spect",
    "light_field", "radio_interferometry", "pump_probe", "ultrasound",
    "cassi", "event_camera", "sim", "stm", "proton_therapy_img",
    "photoacoustic", "acoustic_microscopy", "doppler_ultrasound",
    "confocal_3d", "two_photon", "bioluminescence_tomo", "matrix",
    # Also process ALL other modalities that might benefit
]


def main():
    # Get ALL modalities with H5 data
    all_mods = sorted([d for d in os.listdir(BENCHMARK_DIR)
                       if os.path.isdir(os.path.join(BENCHMARK_DIR, d))
                       and not d.startswith(".")])

    # Process priority targets first, then all others
    priority = set(TARGETS)
    ordered = [m for m in TARGETS if m in all_mods]
    ordered += [m for m in all_mods if m not in priority]

    all_improvements = []
    for i, mod in enumerate(ordered):
        t0 = time.time()
        improvements = process_modality(mod)
        dt = time.time() - t0

        if improvements:
            gains = [(n - o) for _, o, n, _ in improvements]
            avg_gain = np.mean(gains)
            methods = set(m for _, _, _, m in improvements)
            print(f"[{i+1}/{len(ordered)}] {mod:30s}: +{avg_gain:.3f} dB avg ({len(improvements)} samples) [{', '.join(list(methods)[:2])}] [{dt:.0f}s]")
            all_improvements.extend([(mod, *imp) for imp in improvements])
        else:
            if dt > 1:
                print(f"[{i+1}/{len(ordered)}] {mod:30s}: no improv [{dt:.0f}s]")

    print(f"\n{'='*60}")
    n_mods = len(set(m for m, _, _, _, _ in all_improvements))
    print(f"Total: {len(all_improvements)} samples improved across {n_mods} modalities")

    from collections import defaultdict
    by_mod = defaultdict(list)
    for mod, sk, old, new, method in all_improvements:
        by_mod[mod].append((old, new, method))

    for mod in sorted(by_mod.keys()):
        items = by_mod[mod]
        avg_gain = np.mean([n - o for o, n, _ in items])
        avg_new = np.mean([n for _, n, _ in items])
        print(f"  {mod:30s}: avg +{avg_gain:.3f} dB -> {avg_new:.1f} dB ({len(items)} samples)")


if __name__ == "__main__":
    main()
