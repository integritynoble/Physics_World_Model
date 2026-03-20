#!/usr/bin/env python3
"""Quick test: verify 3D lensless improvement with independent depth PSFs."""
import numpy as np
from scipy import fft as sp_fft
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.data import camera, astronaut
from skimage.transform import resize
from skimage.restoration import denoise_tv_chambolle
import time
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "papers/system_design/expert_study")
from run_new_modalities import (
    generate_depth_phase_psfs, fft_convolve2d,
    gap_tv, fista_tv, admm_tv, rl_deconv,
)

N = 128
NZ = 8
N_SAMPLES = 3

def generate_test_images(n_samples, size):
    images = []
    imgs_raw = [camera(), astronaut()[:, :, 0],
                resize(np.random.RandomState(99).rand(64, 64), (size, size))]
    for i in range(n_samples):
        img = resize(imgs_raw[i % len(imgs_raw)], (size, size)).astype(np.float64)
        img = img / (img.max() + 1e-10)
        images.append(img)
    return images

def run_3d_lensless():
    print("Generating depth-dependent PSFs (independent phases)...", flush=True)
    psfs = generate_depth_phase_psfs(NZ, N, seed=42, feature_scale=2.5)

    # Check PSF diversity
    corrs = []
    for i in range(NZ):
        for j in range(i + 1, NZ):
            c = np.corrcoef(psfs[i].ravel(), psfs[j].ravel())[0, 1]
            corrs.append(c)
    print(f"  PSF cross-correlation: mean={np.mean(corrs):.4f}, "
          f"max={np.max(corrs):.4f}, min={np.min(corrs):.4f}", flush=True)

    images = generate_test_images(N_SAMPLES, N)

    # Precompute FFTs
    H = [sp_fft.fft2(psf, s=(N, N)) for psf in psfs]
    Hconj = [np.conj(h) for h in H]
    eps = 1e-3

    for si, img in enumerate(images):
        # Create multi-depth scene
        x_vol = np.zeros((NZ, N, N))
        for z in range(NZ):
            off = z * 7
            x_vol[z] = np.roll(np.roll(img, off, 0), -off, 1) * (0.5 + 0.5 * z / NZ)

        # Forward model: y = sum_z H_z * x_z
        y = np.zeros((N, N))
        for z in range(NZ):
            y += fft_convolve2d(x_vol[z], psfs[z])
        y += np.random.randn(N, N) * 0.01

        # Define forward/adjoint operators
        def forward_fn(x_stack):
            out = np.zeros((N, N))
            for z in range(NZ):
                out += np.real(sp_fft.ifft2(sp_fft.fft2(x_stack[z]) * H[z]))
            return out

        def adjoint_fn(y_in):
            out = np.zeros((NZ, N, N))
            Y = sp_fft.fft2(y_in)
            for z in range(NZ):
                out[z] = np.real(sp_fft.ifft2(Y * Hconj[z]))
            return out

        algos = {
            "GAP-TV": lambda: gap_tv(y, forward_fn, adjoint_fn, NZ, n_iter=80, tv_weight=0.008),
            "FISTA+TV": lambda: fista_tv(y, forward_fn, adjoint_fn, NZ, step=0.5/NZ, n_iter=80, tv_weight=0.006),
            "ADMM+TV": lambda: admm_tv(y, forward_fn, adjoint_fn, NZ, n_iter=60, rho=0.5, tv_weight=0.008),
            "R-L": lambda: rl_deconv(y, forward_fn, adjoint_fn, NZ, n_iter=60),
        }

        for name, algo_fn in algos.items():
            t0 = time.time()
            x_recon = algo_fn()
            dt = time.time() - t0
            p = psnr(x_vol, x_recon, data_range=x_vol.max())
            print(f"  Sample {si}, {name}: PSNR={p:.2f} dB ({dt:.1f}s)", flush=True)

run_3d_lensless()
