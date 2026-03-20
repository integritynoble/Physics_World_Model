#!/usr/bin/env python3
"""Final comprehensive solver verification with all fixes applied."""
import sys, os, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))
sys.path.insert(0, str(ROOT))

# Clear cached modules
for m in list(sys.modules.keys()):
    if 'pwm_core' in m:
        del sys.modules[m]

rng = np.random.RandomState(42)
N = 64
results = {}

def make_phantom(N=64):
    gt = np.zeros((N, N), dtype=np.float32)
    y, x = np.mgrid[-1:1:N*1j, -1:1:N*1j]
    gt += 1.0 * ((x/0.69)**2 + (y/0.92)**2 < 1)
    gt -= 0.8 * ((x/0.66)**2 + (y/0.88)**2 < 1)
    return gt

def make_blobs(N=64, n=10):
    gt = np.zeros((N, N), dtype=np.float32)
    for _ in range(n):
        cx, cy = rng.randint(5, N-5, 2)
        r2 = rng.randint(2, 6)
        yy, xx = np.ogrid[-cx:N-cx, -cy:N-cy]
        gt[xx**2 + yy**2 <= r2**2] = rng.uniform(0.3, 1.0)
    return gt

def test(name, fn):
    t0 = time.time()
    try:
        out = fn()
        t = time.time() - t0
        out = np.real(np.asarray(out)).astype(np.float32)
        print(f"  {name:50s} OK  {str(out.shape):15s} t={t:.2f}s")
        results[name] = "OK"
    except Exception as e:
        t = time.time() - t0
        print(f"  {name:50s} ERR {str(e)[:50]}")
        results[name] = "ERR"

print("=" * 70)
print("FINAL COMPREHENSIVE SOLVER VERIFICATION")
print("=" * 70)

from scipy.signal import fftconvolve
from scipy.ndimage import rotate

gt = make_phantom(N)
psf = np.zeros((15,15), dtype=np.float32)
c2 = 7
for yi in range(15):
    for xi in range(15):
        psf[yi,xi] = np.exp(-((xi-c2)**2+(yi-c2)**2)/8)
psf /= psf.sum()

# ─── CT ───────────────────────────────────────────────────────────────
print("\n--- CT ---")
from pwm_core.recon.ct_solvers import fbp_2d, sart_2d, run_fbp, run_sart
angs = np.linspace(0, np.pi, 30, endpoint=False)
sino = np.zeros((30, N), dtype=np.float32)
for i, a in enumerate(angs):
    sino[i] = rotate(gt, np.degrees(a), reshape=False).sum(axis=0)

class CTOp:
    def __init__(self):
        self.angles = angs; self.x_shape = (N, N)
    def forward(self, x): return x
    def adjoint(self, y): return y

test("CT fbp_2d (CPU)", lambda: fbp_2d(sino, angs, "ramlak", N))
test("CT sart_2d (CPU)", lambda: sart_2d(sino, angs, N, iterations=5))
test("CT run_fbp wrapper (CPU)", lambda: run_fbp(sino, CTOp(), {"filter":"ramlak","output_size":N})[0])
test("CT run_sart wrapper (CPU)", lambda: run_sart(sino, CTOp(), {"iterations":5,"output_size":N})[0])
test("CT fbp_2d (GPU)", lambda: fbp_2d(sino, angs, "ramlak", N, device="cuda"))

# ─── MRI ──────────────────────────────────────────────────────────────
print("\n--- MRI ---")
from pwm_core.recon.mri_solvers import zero_filled_reconstruction, cs_mri_wavelet
gt_mri = make_phantom(N)
kspace = np.fft.fftshift(np.fft.fft2(gt_mri))
mask = np.zeros((N,N), dtype=bool)
cc = N//2; mask[:, cc-3:cc+3] = True
for j in range(N):
    if not mask[0,j] and rng.random()<0.25: mask[:,j] = True
ks = (kspace * mask).astype(np.complex64)

test("MRI zero_filled (CPU)", lambda: zero_filled_reconstruction(ks, mask, device="cpu"))
test("MRI cs_wavelet (CPU)", lambda: cs_mri_wavelet(ks, mask.astype(np.float32), device="cpu"))
test("MRI zero_filled (GPU)", lambda: zero_filled_reconstruction(ks, mask, device="cuda"))
test("MRI cs_wavelet (GPU)", lambda: cs_mri_wavelet(ks, mask.astype(np.float32), device="cuda"))

# ─── Richardson-Lucy ──────────────────────────────────────────────────
print("\n--- Richardson-Lucy ---")
from pwm_core.recon.richardson_lucy import richardson_lucy_2d, run_richardson_lucy
gt_rl = make_blobs(N)
y_rl = np.maximum(fftconvolve(gt_rl, psf, mode="same") + rng.normal(0,0.01,(N,N)).astype(np.float32), 1e-6)

class RLOp:
    def __init__(self):
        self.x_shape = (N,N); self.psf = psf
    def forward(self, x): return fftconvolve(x, self.psf, mode="same").astype(np.float32)
    def adjoint(self, y): return fftconvolve(y, self.psf[::-1,::-1], mode="same").astype(np.float32)
    def info(self): return {"psf": self.psf}

test("RL 2D (CPU)", lambda: richardson_lucy_2d(y_rl, psf, iterations=20))
test("RL wrapper (CPU)", lambda: run_richardson_lucy(y_rl, RLOp(), {"iters":20})[0])

# ─── PnP ──────────────────────────────────────────────────────────────
print("\n--- PnP ---")
from pwm_core.recon.pnp import run_pnp
class IdOp:
    def __init__(self): self.x_shape = (N,N)
    def forward(self, x): return x.copy()
    def adjoint(self, y): return y.copy()

noisy = gt + rng.normal(0, 0.1, gt.shape).astype(np.float32)
test("PnP-HQS NLM (CPU)", lambda: run_pnp(noisy, IdOp(), {"algorithm":"hqs","denoiser":"nlm","iters":5})[0])
test("PnP-HQS Gaussian (CPU)", lambda: run_pnp(noisy, IdOp(), {"algorithm":"hqs","denoiser":"gaussian","iters":5})[0])

# ─── GAP-TV ───────────────────────────────────────────────────────────
print("\n--- GAP-TV ---")
from pwm_core.recon.gap_tv import run_gap_tv
n_bands = 4
gt_cube = rng.rand(N,N,n_bands).astype(np.float32)*0.5+0.3
cmask = (rng.rand(N,N)>0.5).astype(np.float32)
y_cassi = np.zeros((N,N+(n_bands-1)),dtype=np.float32)
for l in range(n_bands):
    y_cassi[:,l:l+N] += gt_cube[:,:,l]*cmask

class CAOp:
    def __init__(self):
        self.mask=cmask; self.n_bands=n_bands; self.step=1; self.x_shape=(N,N,n_bands)
    def forward(self,x): return x
    def adjoint(self,y): return y
    def info(self): return {"modality":"cassi","mask":self.mask,"n_bands":self.n_bands,"step":1}

test("GAP-TV CASSI (CPU)", lambda: run_gap_tv(y_cassi, CAOp(), {"iters":5})[0])

T = 8
gt_vid = np.stack([make_blobs(N,5+t) for t in range(T)]).astype(np.float32)
masks_c = (rng.rand(T,N,N)>0.5).astype(np.float32)
y_cacti = np.sum(gt_vid*masks_c, axis=0)

class CAcOp:
    def __init__(self):
        self.masks=masks_c; self.n_frames=T; self.x_shape=(T,N,N)
    def forward(self,x): return np.sum(x*self.masks, axis=0)
    def adjoint(self,y): return self.masks*y[None,:,:]
    def info(self): return {"modality":"cacti","masks":self.masks,"n_frames":T}

test("GAP-TV CACTI (CPU)", lambda: run_gap_tv(y_cacti, CAcOp(), {"iters":5})[0])

# ─── SPC ──────────────────────────────────────────────────────────────
print("\n--- SPC ---")
from pwm_core.recon.spc_solvers import admm_spc, fista_l1
gt_spc = make_phantom(32).ravel()
M = 32*32//4
A_spc = rng.randn(M, 32*32).astype(np.float32)/np.sqrt(M)
y_spc = A_spc @ gt_spc + rng.randn(M).astype(np.float32)*0.01

test("SPC ADMM (CPU)", lambda: admm_spc(y_spc, A_spc, device="cpu"))
test("SPC FISTA-L1 (CPU)", lambda: fista_l1(y_spc, A_spc, device="cpu"))
test("SPC ADMM (GPU)", lambda: admm_spc(y_spc, A_spc, device="cuda"))

# ─── Holography ───────────────────────────────────────────────────────
print("\n--- Holography ---")
from pwm_core.recon.holography_solver import angular_spectrum_propagate
gt_holo = make_blobs(N, 10).astype(np.float32)
field = gt_holo * np.exp(1j * rng.uniform(-0.5, 0.5, (N,N)).astype(np.float32))
holo = angular_spectrum_propagate(field, 532e-9, 5e-6, 1e-3)
test("Holography AS (CPU)", lambda: np.abs(angular_spectrum_propagate(np.sqrt(np.abs(holo)**2+0j), 532e-9, 5e-6, -1e-3)))

# ─── OCT ──────────────────────────────────────────────────────────────
print("\n--- OCT ---")
from pwm_core.recon.oct_solver import fft_recon
gt_oct = make_blobs(N)
spectral = np.fft.fft(gt_oct, axis=0).astype(np.complex64)
test("OCT FFT (CPU)", lambda: fft_recon(spectral))

# ─── FISTA-L2 wrapper (new) ──────────────────────────────────────────
print("\n--- FISTA-L2 wrapper ---")
from pwm_core.recon.classical import run_fista_l2
test("FISTA-L2 wrapper (CPU)", lambda: run_fista_l2(noisy, IdOp(), {"iters":10})[0])

# ─── Noise2Void (fixed alias) ────────────────────────────────────────
print("\n--- Noise2Void ---")
from pwm_core.recon.noise2void import noise2void_denoise
test("N2V denoise (CPU, 2 epochs)", lambda: noise2void_denoise(noisy, IdOp(), {"epochs":2, "device":"cpu"})[0])

# ─── Lightsheet ───────────────────────────────────────────────────────
print("\n--- Lightsheet ---")
from pwm_core.recon.lightsheet_solver import run_lightsheet
class LSOp:
    def __init__(self): self.x_shape=(N,N); self.psf=psf
    def forward(self,x): return fftconvolve(x, self.psf, mode="same").astype(np.float32)
    def adjoint(self,y): return fftconvolve(y, self.psf[::-1,::-1], mode="same").astype(np.float32)
    def info(self): return {"x_shape":(N,N),"stripe_direction":"horizontal"}

y_ls = fftconvolve(gt, psf, mode="same").astype(np.float32)
test("Lightsheet Fourier (CPU)", lambda: run_lightsheet(y_ls, LSOp(), {"solver":"fourier_notch"})[0])
test("Lightsheet Wavelet (CPU)", lambda: run_lightsheet(y_ls, LSOp(), {"solver":"wavelet"})[0])

# ─── Lensless ─────────────────────────────────────────────────────────
print("\n--- Lensless ---")
from pwm_core.recon.lensless_solver import run_lensless
psf_ll = rng.rand(N,N).astype(np.float32); psf_ll /= psf_ll.sum()
y_ll = fftconvolve(gt, psf_ll, mode="same").astype(np.float32)
class LLOp:
    def __init__(self): self.x_shape=(N,N); self.psf=psf_ll
    def forward(self,x): return fftconvolve(x, self.psf, mode="same").astype(np.float32)
    def adjoint(self,y): return fftconvolve(y, self.psf[::-1,::-1], mode="same").astype(np.float32)
    def info(self): return {"psf": self.psf}

test("Lensless Wiener (CPU)", lambda: run_lensless(y_ll, LLOp(), {"method":"wiener","reg":0.01})[0])

# ─── SIM ──────────────────────────────────────────────────────────────
print("\n--- SIM ---")
from pwm_core.recon.sim_solver import run_sim_reconstruction
gt_sim = make_blobs(N)
sim_imgs = np.stack([gt_sim + 0.1*rng.randn(N,N).astype(np.float32) for _ in range(9)])
class SIMOp:
    def __init__(self): self.x_shape=(N,N); self.n_orientations=3; self.n_phases=3
    def forward(self,x): return x
    def adjoint(self,y): return y
    def info(self): return {"n_orientations":3,"n_phases":3,"x_shape":(N,N)}

test("SIM reconstruction (CPU)", lambda: run_sim_reconstruction(sim_imgs, SIMOp(), {"n_orientations":3,"n_phases":3})[0])

# ─── NeRF ─────────────────────────────────────────────────────────────
print("\n--- NeRF ---")
from pwm_core.recon.nerf_solver import run_nerf
views = np.stack([gt + rng.normal(0,0.05,(N,N)).astype(np.float32) for _ in range(4)])
class NeRFOp:
    def __init__(self): self.x_shape=(N,N); self.n_views=4
    def forward(self,x): return x
    def adjoint(self,y): return y
    def info(self): return {"n_views":4,"x_shape":(N,N)}

test("NeRF fusion (CPU)", lambda: run_nerf(views, NeRFOp(), {"n_views":4,"iters":3})[0])

# ─── Panorama ─────────────────────────────────────────────────────────
print("\n--- Panorama ---")
from pwm_core.recon.panorama_solver import run_panorama_fusion
pano_views = np.stack([gt + rng.normal(0,0.02,(N,N)).astype(np.float32) for _ in range(4)])
class PanoOp:
    def __init__(self): self.x_shape=(N,N); self.n_views=4
    def forward(self,x): return x
    def adjoint(self,y): return y
    def info(self): return {"n_views":4}

test("Panorama fusion (CPU)", lambda: run_panorama_fusion(pano_views, PanoOp(), {"n_views":4})[0])

# ─── Integral ─────────────────────────────────────────────────────────
print("\n--- Integral ---")
from pwm_core.recon.integral_solver import run_integral
int_views = np.stack([gt + rng.normal(0,0.02,(N,N)).astype(np.float32) for _ in range(9)])
class IntOp:
    def __init__(self): self.x_shape=(N,N); self.n_views=9
    def forward(self,x): return x
    def adjoint(self,y): return y
    def info(self): return {"n_views":9}

test("Integral imaging (CPU)", lambda: run_integral(int_views, IntOp(), {"n_views":9})[0])

# ─── FLIM ─────────────────────────────────────────────────────────────
print("\n--- FLIM ---")
from pwm_core.recon.flim_solver import mle_fit_recon, run_flim
gt_flim = make_blobs(N)
n_time = 32
t_arr = np.linspace(0, 10, n_time)
time_data = np.zeros((n_time, N, N), dtype=np.float32)
for i in range(N):
    for j in range(N):
        tau = max(gt_flim[i,j]*3+1, 0.1)
        time_data[:,i,j] = gt_flim[i,j]*np.exp(-t_arr/tau)
time_data += rng.normal(0,0.01,time_data.shape).astype(np.float32)

class FLIMOp:
    def __init__(self): self.x_shape=(N,N); self.time_bins=n_time; self.t=t_arr
    def forward(self,x): return x
    def adjoint(self,y): return y
    def info(self): return {"time_bins":n_time}

test("FLIM MLE (CPU)", lambda: mle_fit_recon(time_data, FLIMOp(), {"time_bins":n_time})[0])

# ─── Ptychography ─────────────────────────────────────────────────────
print("\n--- Ptychography ---")
from pwm_core.recon.ptychography_solver import run_epie
gt_pty = make_blobs(N)
probe_size = 16
scan_pos = [(i*8, j*8) for i in range(4) for j in range(4)]
probe = np.ones((probe_size, probe_size), dtype=np.complex64)
patterns = np.zeros((len(scan_pos), probe_size, probe_size), dtype=np.float32)
for idx, (py, px) in enumerate(scan_pos):
    if py+probe_size<=N and px+probe_size<=N:
        patch = gt_pty[py:py+probe_size, px:px+probe_size]
        dp = np.abs(np.fft.fft2(patch*probe))**2
        patterns[idx] = dp.astype(np.float32)

class PtyOp:
    def __init__(self):
        self.scan_positions=scan_pos; self.probe=probe; self.x_shape=(N,N)
    def forward(self,x): return x
    def adjoint(self,y): return y
    def info(self): return {"scan_positions":self.scan_positions,"probe_size":probe_size}

test("ePIE ptychography (CPU)", lambda: run_epie(patterns, PtyOp(), {"iterations":5})[0])

# ─── Summary ──────────────────────────────────────────────────────────
ok = sum(1 for v in results.values() if v=="OK")
err = sum(1 for v in results.values() if v=="ERR")
print(f"\n{'='*70}")
print(f"SUMMARY: {ok} OK, {err} ERR, {len(results)} total")

import json
with open(ROOT / "benchmark_results" / "final_verification.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to benchmark_results/final_verification.json")
