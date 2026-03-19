#!/usr/bin/env python3
"""CUP forward model simulation and reconstruction from standard images.

CUP (Compressed Ultrafast Photography) uses temporal dispersion identical to
CASSI's spectral dispersion — the streak camera shears time frames spatially:
  y[i, j + t*step] += mask[i, j] * x[i, j, t]

- x: (H, W, T) spatiotemporal scene — T time frames
- mask: (H, W) single binary coded aperture (like CASSI)
- y: (H, W_ext) single 2D measurement where W_ext = W + (T-1)*step

This is identical to SD-CASSI with temporal frames replacing spectral bands.
We use gap_tv_cassi from pwm_core for reconstruction.

References:
- Liang et al. (2018) Science Advances — single-shot CUP at 10^11 fps
- Gao et al. (2014) Nature — CUP at 100 billion fps
- Yuan (2016) — GAP-TV for CASSI/CACTI reconstruction
"""
import os, sys
import numpy as np
import h5py
from pathlib import Path
from scipy.ndimage import gaussian_filter, shift
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add pwm_core to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))

from pwm_core.recon.gap_tv import gap_tv_cassi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STANDARD_DIR = ROOT / "datasets" / "benchmark" / "cup" / "standard"
OUT_DIR = STANDARD_DIR / "images" / "cup_simulation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def psnr(gt, pred):
    gt, pred = gt.astype(np.float64), pred.astype(np.float64)
    mse = np.mean((gt - pred) ** 2)
    if mse < 1e-15:
        return 100.0
    mx = max(np.max(np.abs(gt)), 1e-10)
    return float(10 * np.log10(mx ** 2 / mse))


def ssim_simple(gt, pred):
    """Simple SSIM approximation."""
    gt, pred = gt.astype(np.float64), pred.astype(np.float64)
    mu_g, mu_p = np.mean(gt), np.mean(pred)
    sig_g, sig_p = np.std(gt), np.std(pred)
    sig_gp = np.mean((gt - mu_g) * (pred - mu_p))
    c1 = (0.01 * np.max(gt)) ** 2
    c2 = (0.03 * np.max(gt)) ** 2
    num = (2 * mu_g * mu_p + c1) * (2 * sig_gp + c2)
    den = (mu_g**2 + mu_p**2 + c1) * (sig_g**2 + sig_p**2 + c2)
    return float(num / den)


def create_temporal_scene(img_2d, n_frames=8, scene_type="moving"):
    """Create a 3D (H, W, T) spatiotemporal scene from a 2D image.

    Simulates ultrafast phenomena captured by CUP.
    """
    H, W = img_2d.shape
    sz = min(H, W, 256)
    r0 = (H - sz) // 2
    c0 = (W - sz) // 2
    crop = img_2d[r0:r0+sz, c0:c0+sz].copy().astype(np.float64)
    crop = (crop - crop.min()) / (crop.max() - crop.min() + 1e-10)

    scene = np.zeros((sz, sz, n_frames), dtype=np.float64)

    if scene_type == "moving":
        # Moving wavefront / shifting object
        for t in range(n_frames):
            dx = (t - n_frames // 2) * 3.0
            dy = (t - n_frames // 2) * 2.0
            shifted = shift(crop, [dy, dx], mode='reflect')
            sigma = 0.5 + 0.2 * t
            scene[:, :, t] = gaussian_filter(shifted, sigma=sigma)
    elif scene_type == "expanding":
        # Expanding ring / pulse (like laser-matter interaction)
        cy, cx = sz // 2, sz // 2
        yy, xx = np.mgrid[:sz, :sz]
        for t in range(n_frames):
            radius = 15 + t * 12
            ring = np.exp(-((np.sqrt((yy - cy)**2 + (xx - cx)**2) - radius)**2) / (2 * 8**2))
            scene[:, :, t] = crop * (0.2 + 0.8 * ring)
    elif scene_type == "pulsing":
        # Pulsing intensity (like fluorescence lifetime)
        for t in range(n_frames):
            phase = 2 * np.pi * t / n_frames
            modulation = 0.4 + 0.6 * np.sin(phase)
            blurred = gaussian_filter(crop, sigma=0.5 + 2.0 * abs(np.sin(phase)))
            scene[:, :, t] = blurred * modulation

    scene = np.clip(scene, 0, None)
    scene = scene / (scene.max() + 1e-10)
    return scene.astype(np.float32)


def cup_forward(x, mask, step=2):
    """CUP forward with temporal dispersion (identical to CASSI).

    y[i, j + t*step] += mask[i, j] * x[i, j, t]
    """
    H, W, T = x.shape
    W_ext = W + (T - 1) * step
    y = np.zeros((H, W_ext), dtype=np.float64)
    for t in range(T):
        y[:, t*step:t*step+W] += mask * x[:, :, t]
    return y.astype(np.float32)


def cup_adjoint(y, mask, T, step=2):
    """CUP adjoint (back-projection with dispersion).

    x[i, j, t] = mask[i, j] * y[i, j + t*step]
    """
    H = mask.shape[0]
    W = mask.shape[1]
    x = np.zeros((H, W, T), dtype=np.float32)
    for t in range(T):
        x[:, :, t] = mask * y[:, t*step:t*step+W]
    return x


def main():
    print("=" * 70)
    print("CUP FORWARD MODEL (DISPERSION-BASED) & GAP-TV RECONSTRUCTION")
    print("Using pwm_core.recon.gap_tv.gap_tv_cassi (temporal = spectral)")
    print("=" * 70)

    h5_files = sorted(STANDARD_DIR.glob("standard_cup_*.h5"))
    print(f"Found {len(h5_files)} standard H5 files")

    N_FRAMES = 8
    STEP = 2  # dispersion step: 2 pixels per frame (same as CASSI)

    scene_types = ["moving", "expanding", "pulsing"]
    all_results = []

    for idx, h5_path in enumerate(h5_files[:5]):
        print(f"\n{'='*60}")
        print(f"Sample {idx}: {h5_path.name}")

        with h5py.File(str(h5_path), "r") as f:
            img = f["x_true"][:]
        print(f"  Source image: {img.shape}, range [{img.min():.3f}, {img.max():.3f}]")

        stype = scene_types[idx % len(scene_types)]
        print(f"  Scene type: {stype}")

        # Create 3D temporal scene
        x_true = create_temporal_scene(img, n_frames=N_FRAMES, scene_type=stype)
        H, W, T = x_true.shape
        print(f"  x_true: {x_true.shape}, max={x_true.max():.3f}")

        # Single 2D binary coded aperture mask (same as CASSI)
        rng = np.random.RandomState(42 + idx)
        mask = (rng.rand(H, W) < 0.5).astype(np.float32)

        # CUP forward model with temporal dispersion
        y = cup_forward(x_true, mask, step=STEP)
        noise_std = 0.01
        y_noisy = y + noise_std * rng.standard_normal(y.shape).astype(np.float32)
        y_noisy = np.clip(y_noisy, 0, None)
        W_ext = W + (T - 1) * STEP
        print(f"  y: {y_noisy.shape} (W_ext={W_ext}), range [{y_noisy.min():.3f}, {y_noisy.max():.3f}]")

        # 1) Naive adjoint (back-projection, no normalization)
        x_adj = cup_adjoint(y_noisy, mask, T, step=STEP)
        adj_psnr = psnr(x_true, x_adj)
        print(f"  Adjoint (naive): PSNR={adj_psnr:.2f} dB")

        # 2) GAP-TV via pwm_core (same as CASSI, treating time as spectral)
        print(f"  GAP-TV (iters=100, lam=0.05, step={STEP})...")
        x_gap = gap_tv_cassi(y_noisy, mask, n_bands=N_FRAMES,
                             iterations=100, lam=0.05, step=STEP, device='cpu')
        gap_psnr = psnr(x_true, x_gap)
        gap_ssim = ssim_simple(x_true, x_gap)
        print(f"  GAP-TV: PSNR={gap_psnr:.2f} dB, SSIM={gap_ssim:.4f}")

        # 3) GAP-TV with more iterations and finer lambda
        print(f"  GAP-TV-fine (iters=200, lam=0.03)...")
        x_gap2 = gap_tv_cassi(y_noisy, mask, n_bands=N_FRAMES,
                              iterations=200, lam=0.03, step=STEP, device='cpu')
        gap2_psnr = psnr(x_true, x_gap2)
        gap2_ssim = ssim_simple(x_true, x_gap2)
        print(f"  GAP-TV-fine: PSNR={gap2_psnr:.2f} dB, SSIM={gap2_ssim:.4f}")

        # 4) GAP-TV with accelerated (Nesterov) convergence
        print(f"  GAP-TV-accel (iters=200, lam=0.05, accelerated)...")
        x_gap3 = gap_tv_cassi(y_noisy, mask, n_bands=N_FRAMES,
                              iterations=200, lam=0.05, step=STEP,
                              accelerate=True, device='cpu')
        gap3_psnr = psnr(x_true, x_gap3)
        gap3_ssim = ssim_simple(x_true, x_gap3)
        print(f"  GAP-TV-accel: PSNR={gap3_psnr:.2f} dB, SSIM={gap3_ssim:.4f}")

        # Pick best reconstruction
        results = [
            ("GAP-TV", gap_psnr, gap_ssim, x_gap),
            ("GAP-TV-fine", gap2_psnr, gap2_ssim, x_gap2),
            ("GAP-TV-accel", gap3_psnr, gap3_ssim, x_gap3),
        ]
        best_name, best_psnr, best_ssim, best_recon = max(results, key=lambda r: r[1])
        print(f"  Best: {best_name} ({best_psnr:.2f} dB)")

        all_results.append({
            "sample": idx, "scene_type": stype,
            "adj_psnr": round(adj_psnr, 2),
            "gap_psnr": round(gap_psnr, 2), "gap_ssim": round(gap_ssim, 4),
            "gap2_psnr": round(gap2_psnr, 2),
            "gap3_psnr": round(gap3_psnr, 2),
            "best": best_name, "best_psnr": round(best_psnr, 2),
            "best_ssim": round(best_ssim, 4),
        })

        # ---- Visualization: full comparison ----
        vmax = np.max(x_true)
        fig, axes = plt.subplots(4, N_FRAMES, figsize=(3 * N_FRAMES, 12))
        fig.suptitle(
            f"CUP Sample {idx} ({stype}) — Temporal Dispersion step={STEP}\n"
            f"Adjoint: {adj_psnr:.1f} dB | {best_name}: {best_psnr:.1f} dB, "
            f"SSIM={best_ssim:.3f}",
            fontsize=14, fontweight='bold'
        )

        for t in range(N_FRAMES):
            axes[0, t].imshow(x_true[:, :, t], cmap='inferno', vmin=0, vmax=vmax)
            axes[0, t].set_title(f't={t}', fontsize=10)
            axes[0, t].axis('off')

            axes[1, t].imshow(x_adj[:, :, t], cmap='inferno', vmin=0, vmax=vmax)
            axes[1, t].axis('off')

            axes[2, t].imshow(best_recon[:, :, t], cmap='inferno', vmin=0, vmax=vmax)
            axes[2, t].axis('off')

            diff = np.abs(x_true[:, :, t] - best_recon[:, :, t])
            axes[3, t].imshow(diff, cmap='magma', vmin=0, vmax=vmax * 0.3)
            axes[3, t].axis('off')

        labels = ['Ground Truth', 'Adjoint (naive)', f'{best_name} Recon',
                  f'|Error| ({best_name})']
        for r, label in enumerate(labels):
            axes[r, 0].set_ylabel(label, fontsize=11, fontweight='bold',
                                  rotation=90, labelpad=10)
            axes[r, 0].yaxis.set_visible(True)
            axes[r, 0].tick_params(left=False, labelleft=False)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fname = OUT_DIR / f"cup_dispersion_sample_{idx:02d}.png"
        plt.savefig(str(fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

        # ---- Pipeline diagram ----
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
        fig.suptitle(
            f"CUP Pipeline — Sample {idx} ({stype})\n"
            f"Temporal dispersion = CASSI spectral dispersion",
            fontsize=13, fontweight='bold'
        )

        axes[0].imshow(x_true[:, :, N_FRAMES//2], cmap='inferno')
        axes[0].set_title(f"Scene (t={N_FRAMES//2})\n{H}x{W}x{T} volume")
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f"Coded Aperture\n{H}x{W} binary mask")
        axes[1].axis('off')

        axes[2].imshow(y_noisy, cmap='inferno', aspect='auto')
        axes[2].set_title(f"Measurement\n{H}x{W_ext} (dispersed)")
        axes[2].axis('off')

        axes[3].imshow(x_adj[:, :, N_FRAMES//2], cmap='inferno', vmin=0, vmax=vmax)
        axes[3].set_title(f"Adjoint\nPSNR={adj_psnr:.1f} dB")
        axes[3].axis('off')

        axes[4].imshow(best_recon[:, :, N_FRAMES//2], cmap='inferno', vmin=0, vmax=vmax)
        axes[4].set_title(f"{best_name}\nPSNR={best_psnr:.1f} dB")
        axes[4].axis('off')

        plt.tight_layout()
        fname2 = OUT_DIR / f"cup_dispersion_pipeline_{idx:02d}.png"
        plt.savefig(str(fname2), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname2}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY — CUP with temporal dispersion (CASSI-style GAP-TV)")
    print(f"{'='*70}")
    print(f"{'Sample':>8} {'Type':>10} {'Adj':>8} {'GAP-TV':>8} "
          f"{'Fine':>8} {'Accel':>8} {'Best':>12} {'PSNR':>8}")
    for r in all_results:
        print(f"{r['sample']:>8} {r['scene_type']:>10} {r['adj_psnr']:>7.1f}  "
              f"{r['gap_psnr']:>7.1f}  {r['gap2_psnr']:>7.1f}  "
              f"{r['gap3_psnr']:>7.1f}  {r['best']:>12} {r['best_psnr']:>7.1f}")
    avg_best = np.mean([r['best_psnr'] for r in all_results])
    avg_ssim = np.mean([r['best_ssim'] for r in all_results])
    print(f"\nAverage best: {avg_best:.1f} dB, SSIM={avg_ssim:.4f}")


if __name__ == "__main__":
    main()
