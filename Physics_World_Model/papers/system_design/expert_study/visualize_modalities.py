#!/usr/bin/env python3
"""Generate reconstruction images for all 9 modalities in one folder."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage, fft as sp_fft
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.data import camera, astronaut
from skimage.transform import resize
from skimage.restoration import denoise_tv_chambolle
import os, sys, time, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from run_new_modalities import (
    N, NZ, NT, generate_phase_mask_psf, generate_depth_phase_psfs,
    generate_binary_mask, fft_convolve2d, add_noise,
    wiener_deconv_fft, apply_shift, apply_shift_adjoint,
    get_test_images, generate_3d_volumes, generate_video_frames,
    generate_spectral_cube, generate_dispersion_shifts, generate_streak_shifts,
    gap_tv, fista_tv, admm_tv, rl_deconv,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "modality_images")
os.makedirs(OUT_DIR, exist_ok=True)

NB = 8  # spectral bands for spectral modality


def save_2d(name, gt, measurement, recon, p):
    """Save ground truth / measurement / reconstruction side-by-side for 2D."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth')
    axes[1].imshow(measurement, cmap='gray')
    axes[1].set_title('Measurement')
    axes[2].imshow(recon, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Reconstruction\nPSNR={p:.1f} dB')
    for ax in axes:
        ax.axis('off')
    fig.suptitle(name, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name.replace(' ', '_').lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}: PSNR={p:.1f} dB")


def save_3d(name, gt_vol, measurement, recon_vol, p, n_show=4):
    """Save multi-slice: top row GT, bottom row recon."""
    n_slices = gt_vol.shape[0]
    show_idx = np.linspace(0, n_slices - 1, min(n_show, n_slices), dtype=int)
    ncols = len(show_idx) + 1  # +1 for measurement
    fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6))

    # Measurement in first column
    axes[0, 0].imshow(measurement, cmap='gray')
    axes[0, 0].set_title('Measurement')
    axes[1, 0].axis('off')

    for col, idx in enumerate(show_idx):
        axes[0, col + 1].imshow(gt_vol[idx], cmap='gray', vmin=0, vmax=1)
        axes[0, col + 1].set_title(f'GT slice {idx}')
        axes[1, col + 1].imshow(recon_vol[idx], cmap='gray', vmin=0, vmax=1)
        axes[1, col + 1].set_title(f'Recon slice {idx}')

    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle(f'{name} — PSNR={p:.1f} dB', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name.replace(' ', '_').lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}: PSNR={p:.1f} dB")


# ============================================================================
# 1. Lensless (2D)
# ============================================================================
def vis_lensless():
    psf = generate_phase_mask_psf(N, seed=42)
    H_fft = sp_fft.fft2(psf)
    HT_fft = np.conj(H_fft)
    img = get_test_images(1, N)[0]
    y = add_noise(fft_convolve2d(img, psf))

    # ADMM+TV (best from results)
    Y = sp_fft.fft2(y)
    x = np.real(sp_fft.ifft2(HT_fft * Y))
    z = x.copy()
    u = np.zeros_like(x)
    denom = np.abs(H_fft) ** 2 + 1.0
    for _ in range(60):
        rhs = HT_fft * Y + sp_fft.fft2(z - u)
        x = np.real(sp_fft.ifft2(rhs / denom))
        z = denoise_tv_chambolle(np.clip(x + u, 0, None), weight=0.003)
        u = u + x - z
    recon = np.clip(z, 0, 1)
    p = psnr(img, recon, data_range=1.0)
    save_2d("1 Lensless 2D", img, y, recon, p)


# ============================================================================
# 2. 3D Lensless
# ============================================================================
def vis_3d_lensless():
    psfs = generate_depth_phase_psfs(NZ, N)
    H = [sp_fft.fft2(p, s=(N, N)) for p in psfs]
    HT = [np.conj(h) for h in H]
    volumes = generate_3d_volumes(NZ, N, 1)
    vol = volumes[0]

    def forward(v):
        out = np.zeros((N, N))
        for z in range(NZ):
            out += np.real(sp_fft.ifft2(H[z] * sp_fft.fft2(v[z])))
        return out / NZ

    def adjoint(r):
        R = sp_fft.fft2(r)
        out = np.zeros((NZ, N, N))
        for z in range(NZ):
            out[z] = np.real(sp_fft.ifft2(HT[z] * R)) / NZ
        return out

    y = add_noise(forward(vol))
    recon = gap_tv(y, forward, adjoint, NZ, n_iter=100, tv_weight=0.006)
    p = np.mean([psnr(vol[z], recon[z], data_range=1.0) for z in range(NZ)])
    save_3d("2 3D Lensless", vol, y, recon, p)


# ============================================================================
# 3. Temporal-Coded
# ============================================================================
def vis_temporal_coded():
    psf = generate_phase_mask_psf(N, seed=50)
    H_fft = sp_fft.fft2(psf, s=(N, N))
    HT_fft = np.conj(H_fft)
    masks = [generate_binary_mask(N, fill_factor=0.5, seed=100 + t) for t in range(NT)]
    videos = generate_video_frames(NT, N, 1)
    vid = videos[0]

    def forward(v):
        out = np.zeros((N, N))
        for t in range(NT):
            out += np.real(sp_fft.ifft2(H_fft * sp_fft.fft2(masks[t] * v[t])))
        return out / NT

    def adjoint(r):
        R = sp_fft.fft2(r)
        base = np.real(sp_fft.ifft2(HT_fft * R)) / NT
        return np.array([masks[t] * base for t in range(NT)])

    y = add_noise(forward(vid))
    recon = gap_tv(y, forward, adjoint, NT, n_iter=100, tv_weight=0.006)
    p = np.mean([psnr(vid[t], recon[t], data_range=1.0) for t in range(NT)])
    save_3d("3 Temporal-Coded", vid, y, recon, p)


# ============================================================================
# 4. Spectral Lensless
# ============================================================================
def vis_spectral():
    psf = generate_phase_mask_psf(N, seed=60)
    H_fft = sp_fft.fft2(psf, s=(N, N))
    HT_fft = np.conj(H_fft)
    mask = generate_binary_mask(N, fill_factor=0.5, seed=200)
    shifts = generate_dispersion_shifts(NB)
    cubes = generate_spectral_cube(NB, N, 1)
    cube = cubes[0]

    def forward(c):
        out = np.zeros((N, N))
        for b in range(NB):
            shifted = apply_shift(mask * c[b], *shifts[b])
            out += np.real(sp_fft.ifft2(H_fft * sp_fft.fft2(shifted)))
        return out / NB

    def adjoint(r):
        R = sp_fft.fft2(r)
        base = np.real(sp_fft.ifft2(HT_fft * R)) / NB
        return np.array([mask * apply_shift_adjoint(base, *shifts[b]) for b in range(NB)])

    y = add_noise(forward(cube))
    recon = gap_tv(y, forward, adjoint, NB, n_iter=100, tv_weight=0.006)
    p = np.mean([psnr(cube[b], recon[b], data_range=1.0) for b in range(NB)])
    save_3d("4 Spectral Lensless", cube, y, recon, p)


# ============================================================================
# 5. 4D Spectral-Depth
# ============================================================================
def vis_4d_spectral_depth():
    nz, nl = 4, 4
    n_slices = nz * nl
    psfs = generate_depth_phase_psfs(nz, N, seed=700)
    H = [sp_fft.fft2(p, s=(N, N)) for p in psfs]
    HT = [np.conj(h) for h in H]
    shifts = generate_dispersion_shifts(nl, max_shift_px=12)

    from run_new_modalities import _generate_5d
    # Use 4D subset: generate (nz*nl, N, N)
    rng = np.random.RandomState(800)
    cube = np.zeros((n_slices, N, N))
    n_obj = 5
    for j in range(n_obj):
        z = rng.randint(0, nz)
        cx, cy = rng.randint(15, N - 15, 2)
        rx, ry = rng.randint(6, 18, 2)
        yy, xx = np.mgrid[0:N, 0:N]
        spatial = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) + (yy - cy) ** 2 / (2 * ry ** 2)))
        spectrum = ndimage.gaussian_filter1d(rng.rand(nl), sigma=0.8)
        spectrum /= max(spectrum.max(), 1e-10)
        for b in range(nl):
            idx = z * nl + b
            cube[idx] += spatial * spectrum[b] * rng.uniform(0.3, 0.9)
    for idx in range(n_slices):
        cube[idx] += ndimage.gaussian_filter(rng.rand(N, N) * 0.02, sigma=3)
        mx = cube[idx].max()
        if mx > 1e-10:
            cube[idx] = np.clip(cube[idx] / mx, 0, 1)

    def forward(c):
        out = np.zeros((N, N))
        for z in range(nz):
            for b in range(nl):
                idx = z * nl + b
                shifted = apply_shift(c[idx], *shifts[b])
                out += np.real(sp_fft.ifft2(H[z] * sp_fft.fft2(shifted)))
        return out / n_slices

    def adjoint(r):
        R = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(nz):
            base = np.real(sp_fft.ifft2(HT[z] * R)) / n_slices
            for b in range(nl):
                out[z * nl + b] = apply_shift_adjoint(base, *shifts[b])
        return out

    y = add_noise(forward(cube))
    recon = gap_tv(y, forward, adjoint, n_slices, n_iter=80, tv_weight=0.010)
    p = np.mean([psnr(cube[s], recon[s], data_range=1.0) for s in range(n_slices)])
    save_3d("5 4D Spectral-Depth", cube, y, recon, p, n_show=6)


# ============================================================================
# 6. 4D Temporal DMD
# ============================================================================
def vis_4d_temporal_dmd():
    nz, nt = 4, 4
    n_slices = nz * nt
    psfs = generate_depth_phase_psfs(nz, N, seed=900)
    H = [sp_fft.fft2(p, s=(N, N)) for p in psfs]
    HT = [np.conj(h) for h in H]
    masks = [generate_binary_mask(N, fill_factor=0.5, seed=300 + t) for t in range(nt)]

    rng = np.random.RandomState(1000)
    cube = np.zeros((n_slices, N, N))
    n_obj = 5
    for j in range(n_obj):
        z = rng.randint(0, nz)
        base_cx, base_cy = rng.randint(15, N - 15, 2)
        rx, ry = rng.randint(6, 18, 2)
        for t in range(nt):
            cx = int(base_cx + 8 * np.sin(2 * np.pi * t / nt + j)) % N
            cy = int(base_cy + 6 * np.cos(2 * np.pi * t / nt + j * 0.7)) % N
            yy, xx = np.mgrid[0:N, 0:N]
            obj = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) + (yy - cy) ** 2 / (2 * ry ** 2)))
            cube[z * nt + t] += obj * rng.uniform(0.3, 0.9)
    for idx in range(n_slices):
        cube[idx] += ndimage.gaussian_filter(rng.rand(N, N) * 0.02, sigma=3)
        mx = cube[idx].max()
        if mx > 1e-10:
            cube[idx] = np.clip(cube[idx] / mx, 0, 1)

    def forward(c):
        out = np.zeros((N, N))
        for z in range(nz):
            for t in range(nt):
                idx = z * nt + t
                out += np.real(sp_fft.ifft2(H[z] * sp_fft.fft2(masks[t] * c[idx])))
        return out / n_slices

    def adjoint(r):
        R = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(nz):
            base = np.real(sp_fft.ifft2(HT[z] * R)) / n_slices
            for t in range(nt):
                out[z * nt + t] = masks[t] * base
        return out

    y = add_noise(forward(cube))
    recon = gap_tv(y, forward, adjoint, n_slices, n_iter=80, tv_weight=0.008)
    p = np.mean([psnr(cube[s], recon[s], data_range=1.0) for s in range(n_slices)])
    save_3d("6 4D Temporal DMD", cube, y, recon, p, n_show=6)


# ============================================================================
# 7. 4D Temporal Streak
# ============================================================================
def vis_4d_temporal_streak():
    nz, nt = 4, 4
    n_slices = nz * nt
    psfs = generate_depth_phase_psfs(nz, N, seed=1000)
    H = [sp_fft.fft2(p, s=(N, N)) for p in psfs]
    HT = [np.conj(h) for h in H]
    shifts = generate_streak_shifts(nt, max_shift_px=12)

    rng = np.random.RandomState(1050)
    cube = np.zeros((n_slices, N, N))
    n_obj = 5
    for j in range(n_obj):
        z = rng.randint(0, nz)
        base_cx, base_cy = rng.randint(15, N - 15, 2)
        rx, ry = rng.randint(6, 18, 2)
        for t in range(nt):
            cx = int(base_cx + 8 * np.sin(2 * np.pi * t / nt + j)) % N
            cy = int(base_cy + 6 * np.cos(2 * np.pi * t / nt + j * 0.7)) % N
            yy, xx = np.mgrid[0:N, 0:N]
            obj = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) + (yy - cy) ** 2 / (2 * ry ** 2)))
            cube[z * nt + t] += obj * rng.uniform(0.3, 0.9)
    for idx in range(n_slices):
        cube[idx] += ndimage.gaussian_filter(rng.rand(N, N) * 0.02, sigma=3)
        mx = cube[idx].max()
        if mx > 1e-10:
            cube[idx] = np.clip(cube[idx] / mx, 0, 1)

    def forward(c):
        out = np.zeros((N, N))
        for z in range(nz):
            for t in range(nt):
                idx = z * nt + t
                shifted = apply_shift(c[idx], *shifts[t])
                out += np.real(sp_fft.ifft2(H[z] * sp_fft.fft2(shifted)))
        return out / n_slices

    def adjoint(r):
        R = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(nz):
            base = np.real(sp_fft.ifft2(HT[z] * R)) / n_slices
            for t in range(nt):
                out[z * nt + t] = apply_shift_adjoint(base, *shifts[t])
        return out

    y = add_noise(forward(cube))
    recon = gap_tv(y, forward, adjoint, n_slices, n_iter=80, tv_weight=0.010)
    p = np.mean([psnr(cube[s], recon[s], data_range=1.0) for s in range(n_slices)])
    save_3d("7 4D Temporal Streak", cube, y, recon, p, n_show=6)


# ============================================================================
# 8. 5D DMD
# ============================================================================
def vis_5d_dmd():
    nz, nl, nt = 2, 2, 2
    n_slices = nz * nl * nt
    psfs = generate_depth_phase_psfs(nz, N, seed=1100)
    H = [sp_fft.fft2(p, s=(N, N)) for p in psfs]
    HT = [np.conj(h) for h in H]
    shifts = generate_dispersion_shifts(nl, max_shift_px=10)
    masks = [generate_binary_mask(N, fill_factor=0.5, seed=400 + t) for t in range(nt)]

    from run_new_modalities import _generate_5d
    cube = _generate_5d(nz, nl, nt, N, 1)[0]

    def forward(c):
        out = np.zeros((N, N))
        for z in range(nz):
            for b in range(nl):
                for t in range(nt):
                    idx = z * nl * nt + b * nt + t
                    shifted = apply_shift(masks[t] * c[idx], *shifts[b])
                    out += np.real(sp_fft.ifft2(H[z] * sp_fft.fft2(shifted)))
        return out / n_slices

    def adjoint(r):
        R = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(nz):
            base = np.real(sp_fft.ifft2(HT[z] * R)) / n_slices
            for b in range(nl):
                for t in range(nt):
                    idx = z * nl * nt + b * nt + t
                    out[idx] = masks[t] * apply_shift_adjoint(base, *shifts[b])
        return out

    y = add_noise(forward(cube))
    recon = gap_tv(y, forward, adjoint, n_slices, n_iter=80, tv_weight=0.010)
    p = np.mean([psnr(cube[s], recon[s], data_range=1.0) for s in range(n_slices)])
    save_3d("8 5D DMD", cube, y, recon, p, n_show=8)


# ============================================================================
# 9. 5D Streak
# ============================================================================
def vis_5d_streak():
    nz, nl, nt = 2, 2, 2
    n_slices = nz * nl * nt
    psfs = generate_depth_phase_psfs(nz, N, seed=1150)
    H = [sp_fft.fft2(p, s=(N, N)) for p in psfs]
    HT = [np.conj(h) for h in H]
    shifts_l = generate_dispersion_shifts(nl, max_shift_px=10)
    shifts_t = generate_streak_shifts(nt, max_shift_px=10)

    from run_new_modalities import _generate_5d
    cube = _generate_5d(nz, nl, nt, N, 1)[0]

    def forward(c):
        out = np.zeros((N, N))
        for z in range(nz):
            for b in range(nl):
                for t in range(nt):
                    idx = z * nl * nt + b * nt + t
                    dx = shifts_l[b][0] + shifts_t[t][0]
                    dy = shifts_l[b][1] + shifts_t[t][1]
                    shifted = apply_shift(c[idx], dx, dy)
                    out += np.real(sp_fft.ifft2(H[z] * sp_fft.fft2(shifted)))
        return out / n_slices

    def adjoint(r):
        R = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(nz):
            base = np.real(sp_fft.ifft2(HT[z] * R)) / n_slices
            for b in range(nl):
                for t in range(nt):
                    idx = z * nl * nt + b * nt + t
                    dx = shifts_l[b][0] + shifts_t[t][0]
                    dy = shifts_l[b][1] + shifts_t[t][1]
                    out[idx] = apply_shift_adjoint(base, dx, dy)
        return out

    y = add_noise(forward(cube))
    recon = gap_tv(y, forward, adjoint, n_slices, n_iter=80, tv_weight=0.010)
    p = np.mean([psnr(cube[s], recon[s], data_range=1.0) for s in range(n_slices)])
    save_3d("9 5D Streak", cube, y, recon, p, n_show=8)


# ============================================================================
# CASSI (bonus — from expert_reconstructors)
# ============================================================================
def vis_cassi():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "pwm_core"))
    from data_loader import load_data
    from expert_reconstructors import expert_e5_cassi

    samples = load_data("sd_cassi", n_samples=1)
    if not samples:
        print("  CASSI: no data available, skipping")
        return
    s = samples[0]
    x_true = s["x_true"]
    recons = expert_e5_cassi([s])
    recon = recons[0]
    # Show a few spectral bands
    n_bands = x_true.shape[2]
    show_bands = [0, n_bands // 4, n_bands // 2, 3 * n_bands // 4, n_bands - 1]

    fig, axes = plt.subplots(2, len(show_bands), figsize=(3 * len(show_bands), 6))
    for col, b in enumerate(show_bands):
        axes[0, col].imshow(x_true[:, :, b], cmap='gray')
        axes[0, col].set_title(f'GT band {b}')
        axes[1, col].imshow(recon[:, :, b], cmap='gray')
        axes[1, col].set_title(f'Recon band {b}')
    for ax in axes.flat:
        ax.axis('off')

    from skimage.metrics import peak_signal_noise_ratio
    mse = np.mean((x_true - recon) ** 2)
    p = 10 * np.log10(x_true.max() ** 2 / mse) if mse > 1e-10 else 100
    fig.suptitle(f'CASSI (28-band) — PSNR={p:.1f} dB', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "0_cassi_28band.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved CASSI: PSNR={p:.1f} dB")


if __name__ == "__main__":
    np.random.seed(42)
    print("Generating reconstruction images for all modalities...")
    print(f"Output folder: {OUT_DIR}\n")

    t0 = time.time()
    vis_lensless()
    vis_3d_lensless()
    vis_temporal_coded()
    vis_spectral()
    vis_4d_spectral_depth()
    vis_4d_temporal_dmd()
    vis_4d_temporal_streak()
    vis_5d_dmd()
    vis_5d_streak()
    # CASSI needs GCS data — try but don't fail
    try:
        vis_cassi()
    except Exception as e:
        print(f"  CASSI visualization skipped: {e}")

    print(f"\nAll images saved to {OUT_DIR}/ ({time.time() - t0:.0f}s total)")
