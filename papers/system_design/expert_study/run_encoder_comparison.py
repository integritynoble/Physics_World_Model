"""Compare 3 depth encoders × 5 modulation combinations for auto-design study.

Depth Encoders:
  1. Diffuser (Phi_z): Random phase plate + defocus → speckle PSFs
  2. Multi-lens (L_z): Microlens array → parallax sub-pixel shifts
  3. Meta-surface (Psi_z): Engineered phase profile → optimized PSFs

Modulation Combinations:
  A. 3D only (Nz:1)              — depth encoder alone
  B. +Mask (Nz:1)                — coded mask + depth encoder
  C. +M+W_lambda (Nz*Nl:1)       — mask + spectral dispersion + depth
  D. +M+W_t (Nz*Nt:1)           — mask + temporal streak + depth
  E. +M+W_lambda+W_t (Nz*Nl*Nt:1) — mask + spectral + temporal + depth

Total: 3 × 5 = 15 configurations.
"""
import numpy as np
from scipy import ndimage, fft as sp_fft
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
import time
import json
import warnings
warnings.filterwarnings("ignore")

N = 128
N_SAMPLES = 3
NZ = 8    # Depth planes
NL = 4    # Spectral bands
NT = 4    # Temporal frames


# ============================================================================
# Depth Encoder Generators
# ============================================================================

def generate_diffuser_psfs(n_depths, size, seed=42, feature_scale=1.0,
                           defocus_max=40.0):
    """Diffuser-encoded light field: random phase + defocus.

    PSF(z) = |FT{exp(i*phi_random + i*defocus(z)*r^2)}|^2
    """
    rng = np.random.RandomState(seed)
    phase = rng.uniform(0, 2 * np.pi, (size, size))
    if feature_scale > 0:
        phase = ndimage.gaussian_filter(phase, sigma=feature_scale)
    y_arr, x_arr = np.mgrid[-size // 2:size // 2, -size // 2:size // 2]
    r2 = (x_arr ** 2 + y_arr ** 2) / (size / 2.0) ** 2
    H_ffts = []
    for z in range(n_depths):
        dz = z / max(n_depths - 1, 1)
        defocus_phase = defocus_max * dz * np.pi * r2
        field = np.exp(1j * (phase + defocus_phase))
        field = np.fft.ifftshift(field)
        psf = np.abs(sp_fft.fft2(field)) ** 2
        psf /= psf.sum()
        H_ffts.append(sp_fft.fft2(psf))
    return H_ffts


def generate_multilens_psfs(n_depths, size, seed=42, max_parallax=4.0):
    """Multi-lens array: parallax shifts encode depth.

    Each depth plane is shifted by parallax proportional to distance.
    Models a microlens array / light field camera.
    y = sum_z shift(x_z, parallax(z))

    At 128x128 with max_parallax=4 pixels, this is sub-pixel to few-pixel shifts.
    """
    fx = np.fft.fftfreq(size).reshape(1, -1)
    fy = np.fft.fftfreq(size).reshape(-1, 1)
    H_ffts = []
    for z in range(n_depths):
        # Parallax: z=0 has no shift, z=N-1 has max shift
        dz = z / max(n_depths - 1, 1)
        dx = max_parallax * dz
        dy = max_parallax * dz * 0.3  # slight vertical parallax
        # Shift in Fourier domain = phase ramp
        shift_phase = np.exp(-2j * np.pi * (fx * dx + fy * dy))
        H_ffts.append(shift_phase)
    return H_ffts


def generate_metasurface_psfs(n_depths, size, seed=42, defocus_max=40.0):
    """Meta-surface: engineered phase profile + defocus.

    Unlike random diffuser, meta-surface uses deterministic engineered
    phase with azimuthal and radial structure (like a diffractive optical
    element). This creates more structured, deterministic PSFs.

    Phase = a1*r^3 + a2*r*cos(3*theta) + a3*r^2*sin(2*theta)
    (cubic + trefoil + astigmatism — common in engineered meta-surfaces)
    """
    y_arr, x_arr = np.mgrid[-size // 2:size // 2, -size // 2:size // 2]
    r = np.sqrt(x_arr ** 2 + y_arr ** 2) / (size / 2.0)
    r2 = r ** 2
    theta = np.arctan2(y_arr, x_arr)

    # Engineered meta-surface phase (deterministic, not random)
    phase_meta = (8.0 * np.pi * r ** 3                   # cubic (extended DOF)
                  + 6.0 * np.pi * r * np.cos(3 * theta)  # trefoil aberration
                  + 4.0 * np.pi * r2 * np.sin(2 * theta) # astigmatism
                  + 3.0 * np.pi * r * np.sin(5 * theta))  # 5-fold azimuthal

    H_ffts = []
    for z in range(n_depths):
        dz = z / max(n_depths - 1, 1)
        defocus_phase = defocus_max * dz * np.pi * r2
        field = np.exp(1j * (phase_meta + defocus_phase))
        field = np.fft.ifftshift(field)
        psf = np.abs(sp_fft.fft2(field)) ** 2
        psf /= psf.sum()
        H_ffts.append(sp_fft.fft2(psf))
    return H_ffts


# ============================================================================
# Reconstruction Algorithms (shared across all configs)
# ============================================================================

def fista_tv(y, forward_fn, adjoint_fn, n_slices, step, n_iter=150,
             tv_weight=0.003):
    """FISTA with TV regularization — best general-purpose algorithm."""
    x = adjoint_fn(y)
    z_acc = x.copy()
    t = 1.0
    for i in range(n_iter):
        grad = adjoint_fn(forward_fn(z_acc) - y)
        x_new = z_acc - step * grad
        for s in range(n_slices):
            if x_new[s].max() > 0.01:
                x_new[s] = denoise_tv_chambolle(x_new[s], weight=tv_weight)
        x_new = np.clip(x_new, 0, 1)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z_acc = x_new + (t - 1) / t_new * (x_new - x)
        x = x_new
        t = t_new
    return x


def gap_tv(y, forward_fn, adjoint_fn, n_slices, n_iter=120, tv_weight=0.005):
    """GAP-TV: Generalized Alternating Projection with TV denoising."""
    x = adjoint_fn(y)
    for i in range(n_iter):
        residual = y - forward_fn(x)
        x = x + adjoint_fn(residual)
        for s in range(n_slices):
            if x[s].max() > 0.01:
                x[s] = denoise_tv_chambolle(x[s], weight=tv_weight)
        x = np.clip(x, 0, 1)
    return x


# ============================================================================
# Volume Generator (sparse depth — no overlap between depth planes)
# ============================================================================

def generate_sparse_volumes(nz, size, n_samples):
    """3D volumes with objects at specific depths, empty elsewhere."""
    volumes = []
    yy, xx = np.mgrid[0:size, 0:size]
    for s in range(n_samples):
        vol = np.zeros((nz, size, size))
        rng = np.random.RandomState(200 + s)
        n_objects = rng.randint(8, 15)
        for j in range(n_objects):
            z = rng.randint(0, nz)
            cx, cy = rng.randint(15, size - 15, 2)
            rx, ry = rng.randint(8, 30, 2)
            obj = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) +
                           (yy - cy) ** 2 / (2 * ry ** 2)))
            vol[z] += obj * rng.uniform(0.3, 1.0)
        for z in range(nz):
            vol[z] = np.clip(vol[z], 0, 1)
        volumes.append(vol)
    return volumes


def generate_4d_zl(nz, nl, size, n_samples):
    """4D (z, lambda) datacubes. Objects at specific depths with spectra."""
    cubes = []
    yy, xx = np.mgrid[0:size, 0:size]
    for s in range(n_samples):
        cube = np.zeros((nz * nl, size, size))
        rng = np.random.RandomState(800 + s)
        n_obj = rng.randint(6, 12)
        for j in range(n_obj):
            z = rng.randint(0, nz)
            cx, cy = rng.randint(15, size - 15, 2)
            rx, ry = rng.randint(8, 25, 2)
            spatial = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) +
                               (yy - cy) ** 2 / (2 * ry ** 2)))
            spectrum = ndimage.gaussian_filter1d(rng.rand(nl), sigma=0.8)
            spectrum /= max(spectrum.max(), 1e-10)
            for b in range(nl):
                cube[z * nl + b] += spatial * spectrum[b] * rng.uniform(0.3, 1.0)
        for idx in range(nz * nl):
            cube[idx] = np.clip(cube[idx], 0, 1)
        cubes.append(cube)
    return cubes


def generate_4d_zt(nz, nt, size, n_samples):
    """4D (z, t) datacubes. Objects at specific depths with motion."""
    cubes = []
    yy, xx = np.mgrid[0:size, 0:size]
    for s in range(n_samples):
        cube = np.zeros((nz * nt, size, size))
        rng = np.random.RandomState(1000 + s)
        n_obj = rng.randint(4, 8)
        for j in range(n_obj):
            z = rng.randint(0, nz)
            base_cx, base_cy = rng.randint(20, size - 20, 2)
            rx, ry = rng.randint(6, 20, 2)
            amp = rng.uniform(0.3, 1.0)
            for t in range(nt):
                cx = int(base_cx + 8 * np.sin(2 * np.pi * t / nt + j)) % size
                cy = int(base_cy + 5 * np.cos(2 * np.pi * t / nt + j * 0.7)) % size
                obj = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) +
                               (yy - cy) ** 2 / (2 * ry ** 2)))
                cube[z * nt + t] += obj * amp
        for idx in range(nz * nt):
            cube[idx] = np.clip(cube[idx], 0, 1)
        cubes.append(cube)
    return cubes


def generate_5d(nz, nl, nt, size, n_samples):
    """5D (z, lambda, t) datacubes."""
    cubes = []
    yy, xx = np.mgrid[0:size, 0:size]
    for s in range(n_samples):
        n_total = nz * nl * nt
        cube = np.zeros((n_total, size, size))
        rng = np.random.RandomState(1200 + s)
        n_obj = rng.randint(4, 8)
        for j in range(n_obj):
            z = rng.randint(0, nz)
            base_cx, base_cy = rng.randint(15, size - 15, 2)
            rx, ry = rng.randint(6, 18, 2)
            spectrum = ndimage.gaussian_filter1d(rng.rand(nl), sigma=0.8)
            spectrum /= max(spectrum.max(), 1e-10)
            amp = rng.uniform(0.3, 1.0)
            for b in range(nl):
                for t in range(nt):
                    cx = int(base_cx + 6 * np.sin(2 * np.pi * t / nt + j)) % size
                    cy = int(base_cy + 4 * np.cos(2 * np.pi * t / nt + j * 0.7)) % size
                    obj = np.exp(-((xx - cx) ** 2 / (2 * rx ** 2) +
                                   (yy - cy) ** 2 / (2 * ry ** 2)))
                    idx = z * nl * nt + b * nt + t
                    cube[idx] += obj * spectrum[b] * amp
        for idx in range(n_total):
            cube[idx] = np.clip(cube[idx], 0, 1)
        cubes.append(cube)
    return cubes


# ============================================================================
# Utility
# ============================================================================

def generate_binary_mask(size, fill_factor=0.5, seed=None):
    rng = np.random.RandomState(seed)
    return (rng.rand(size, size) < fill_factor).astype(np.float64)


def compute_shift_phase(dx, dy, size):
    fx = np.fft.fftfreq(size).reshape(1, -1)
    fy = np.fft.fftfreq(size).reshape(-1, 1)
    return np.exp(-2j * np.pi * (fx * dx + fy * dy))


def generate_dispersion_shifts(n_bands, max_shift_px=12):
    return [(i * max_shift_px / max(n_bands - 1, 1), 0) for i in range(n_bands)]


# ============================================================================
# Configuration runners
# ============================================================================

def run_3d_only(encoder_name, H_ffts, nz):
    """Config A: 3D depth only (nz:1 compression)."""
    volumes = generate_sparse_volumes(nz, N, N_SAMPLES)

    def forward(vol):
        Y_fft = np.zeros((N, N), dtype=complex)
        for z in range(nz):
            Y_fft += H_ffts[z] * sp_fft.fft2(vol[z])
        return np.real(sp_fft.ifft2(Y_fft)) / nz

    def adjoint(r):
        R_fft = sp_fft.fft2(r)
        out = np.zeros((nz, N, N))
        for z in range(nz):
            out[z] = np.real(sp_fft.ifft2(np.conj(H_ffts[z]) * R_fft))
        return out / nz

    L = max(np.max(np.abs(h) ** 2) for h in H_ffts) / nz + 1e-8
    return _evaluate(f"{encoder_name} 3D-only", volumes, forward, adjoint,
                     nz, 0.8 / L)


def run_3d_mask(encoder_name, H_ffts, nz):
    """Config B: Mask + depth encoder (nz:1 compression)."""
    volumes = generate_sparse_volumes(nz, N, N_SAMPLES)
    masks = [generate_binary_mask(N, 0.5, seed=500 + z) for z in range(nz)]

    def forward(vol):
        Y_fft = np.zeros((N, N), dtype=complex)
        for z in range(nz):
            Y_fft += H_ffts[z] * sp_fft.fft2(masks[z] * vol[z])
        return np.real(sp_fft.ifft2(Y_fft)) / nz

    def adjoint(r):
        R_fft = sp_fft.fft2(r)
        out = np.zeros((nz, N, N))
        for z in range(nz):
            out[z] = masks[z] * np.real(sp_fft.ifft2(np.conj(H_ffts[z]) * R_fft))
        return out / nz

    L = max(np.max(np.abs(h) ** 2) for h in H_ffts) / nz + 1e-8
    return _evaluate(f"{encoder_name} +Mask", volumes, forward, adjoint,
                     nz, 0.8 / L)


def run_4d_mask_spectral(encoder_name, H_ffts_gen, nz, nl=NL):
    """Config C: Mask + spectral dispersion + depth (nz*nl:1)."""
    H_ffts = H_ffts_gen(nz)
    n_slices = nz * nl
    shifts = generate_dispersion_shifts(nl, max_shift_px=12)
    disp_phase = [compute_shift_phase(dx, dy, N) for dx, dy in shifts]
    mask = generate_binary_mask(N, 0.5, seed=750)
    cubes = generate_4d_zl(nz, nl, N, N_SAMPLES)

    def forward(cube):
        Y_fft = np.zeros((N, N), dtype=complex)
        for z in range(nz):
            for b in range(nl):
                idx = z * nl + b
                coded_fft = sp_fft.fft2(mask * cube[idx]) * disp_phase[b]
                Y_fft += H_ffts[z] * coded_fft
        return np.real(sp_fft.ifft2(Y_fft)) / n_slices

    def adjoint(r):
        R_fft = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(nz):
            dec_z = R_fft * np.conj(H_ffts[z])
            for b in range(nl):
                idx = z * nl + b
                out[idx] = mask * np.real(sp_fft.ifft2(
                    dec_z * np.conj(disp_phase[b])))
        return out / n_slices

    L = max(np.max(np.abs(h) ** 2) for h in H_ffts) / n_slices + 1e-8
    return _evaluate(f"{encoder_name} +M+W_l", cubes, forward, adjoint,
                     n_slices, 0.8 / L)


def run_4d_mask_temporal(encoder_name, H_ffts_gen, nz, nt=NT):
    """Config D: Mask + temporal streak + depth (nz*nt:1)."""
    H_ffts = H_ffts_gen(nz)
    n_slices = nz * nt
    masks = [generate_binary_mask(N, 0.5, seed=600 + t) for t in range(nt)]
    cubes = generate_4d_zt(nz, nt, N, N_SAMPLES)

    def forward(cube):
        Y_fft = np.zeros((N, N), dtype=complex)
        for z in range(nz):
            for t in range(nt):
                idx = z * nt + t
                slice_fft = sp_fft.fft2(masks[t] * cube[idx])
                Y_fft += H_ffts[z] * slice_fft
        return np.real(sp_fft.ifft2(Y_fft)) / n_slices

    def adjoint(r):
        R_fft = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(nz):
            dec_z = R_fft * np.conj(H_ffts[z])
            for t in range(nt):
                idx = z * nt + t
                out[idx] = masks[t] * np.real(sp_fft.ifft2(dec_z))
        return out / n_slices

    L = max(np.max(np.abs(h) ** 2) for h in H_ffts) / n_slices + 1e-8
    return _evaluate(f"{encoder_name} +M+W_t", cubes, forward, adjoint,
                     n_slices, 0.8 / L)


def run_5d_full(encoder_name, H_ffts_gen, nz, nl=NL, nt=NT):
    """Config E: Mask + spectral + temporal + depth (nz*nl*nt:1)."""
    H_ffts = H_ffts_gen(nz)
    n_slices = nz * nl * nt
    mask = generate_binary_mask(N, 0.5, seed=750)
    shifts = generate_dispersion_shifts(nl, max_shift_px=12)
    disp_phase = [compute_shift_phase(dx, dy, N) for dx, dy in shifts]
    masks_t = [generate_binary_mask(N, 0.5, seed=600 + t) for t in range(nt)]
    cubes = generate_5d(nz, nl, nt, N, N_SAMPLES)

    def forward(cube):
        Y_fft = np.zeros((N, N), dtype=complex)
        for z in range(nz):
            for b in range(nl):
                for t in range(nt):
                    idx = z * nl * nt + b * nt + t
                    coded = mask * masks_t[t] * cube[idx]
                    coded_fft = sp_fft.fft2(coded) * disp_phase[b]
                    Y_fft += H_ffts[z] * coded_fft
        return np.real(sp_fft.ifft2(Y_fft)) / n_slices

    def adjoint(r):
        R_fft = sp_fft.fft2(r)
        out = np.zeros((n_slices, N, N))
        for z in range(nz):
            dec_z = R_fft * np.conj(H_ffts[z])
            for b in range(nl):
                for t in range(nt):
                    idx = z * nl * nt + b * nt + t
                    out[idx] = mask * masks_t[t] * np.real(sp_fft.ifft2(
                        dec_z * np.conj(disp_phase[b])))
        return out / n_slices

    L = max(np.max(np.abs(h) ** 2) for h in H_ffts) / n_slices + 1e-8
    return _evaluate(f"{encoder_name} +M+W_l+W_t", cubes, forward, adjoint,
                     n_slices, 0.8 / L)


# ============================================================================
# Evaluation
# ============================================================================

def _evaluate(name, data, forward_fn, adjoint_fn, n_slices, step):
    """Run FISTA+TV and GAP-TV, return best PSNR."""
    print(f"\n  {name} ({n_slices} slices -> 1 shot) ...")
    all_psnr_fista = []
    all_psnr_gap = []
    t0 = time.time()

    for s, gt in enumerate(data):
        y = forward_fn(gt)

        # FISTA+TV
        rec_f = fista_tv(y, forward_fn, adjoint_fn, n_slices, step,
                         n_iter=150, tv_weight=0.003)
        p_f = np.mean([psnr(gt[i], rec_f[i], data_range=1.0)
                       for i in range(n_slices)])
        all_psnr_fista.append(p_f)

        # GAP-TV
        rec_g = gap_tv(y, forward_fn, adjoint_fn, n_slices,
                       n_iter=120, tv_weight=0.005)
        p_g = np.mean([psnr(gt[i], rec_g[i], data_range=1.0)
                       for i in range(n_slices)])
        all_psnr_gap.append(p_g)

    t_total = time.time() - t0
    fista_mean = np.mean(all_psnr_fista)
    gap_mean = np.mean(all_psnr_gap)
    best = max(fista_mean, gap_mean)
    best_algo = "FISTA+TV" if fista_mean >= gap_mean else "GAP-TV"
    print(f"    FISTA+TV: {fista_mean:.1f} dB, GAP-TV: {gap_mean:.1f} dB "
          f"-> Best: {best:.1f} dB ({best_algo}) [{t_total:.0f}s]")
    return {
        "fista_psnr": float(fista_mean),
        "gap_psnr": float(gap_mean),
        "best_psnr": float(best),
        "best_algo": best_algo,
        "time_s": float(t_total),
    }


# ============================================================================
# PSF Diversity Analysis
# ============================================================================

def analyze_psf_diversity(name, H_ffts):
    """Compute cross-correlation between PSF pairs."""
    nz = len(H_ffts)
    psfs = [np.real(sp_fft.ifft2(h)) for h in H_ffts]
    psfs = [p / np.linalg.norm(p) for p in psfs]
    corrs = []
    for i in range(nz):
        for j in range(i + 1, nz):
            c = abs(np.sum(psfs[i] * psfs[j]))
            corrs.append(c)
    mean_corr = np.mean(corrs) if corrs else 0
    max_corr = np.max(corrs) if corrs else 0
    print(f"  {name}: PSF cross-corr mean={mean_corr:.4f}, max={max_corr:.4f}")
    return {"mean_corr": float(mean_corr), "max_corr": float(max_corr)}


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("DEPTH ENCODER COMPARISON: Auto-Design Study")
    print("3 encoders × 5 combinations = 15 configurations")
    print("=" * 70)

    # Define encoder generators (lazy — return H_ffts for given nz)
    def diffuser_gen(nz):
        return generate_diffuser_psfs(nz, N, seed=100)

    def multilens_gen(nz):
        return generate_multilens_psfs(nz, N, seed=100)

    def metasurface_gen(nz):
        return generate_metasurface_psfs(nz, N, seed=100)

    encoders = [
        ("Diffuser", diffuser_gen),
        ("Multi-lens", multilens_gen),
        ("Meta-surface", metasurface_gen),
    ]

    # PSF diversity analysis
    print("\n" + "=" * 70)
    print("PSF DIVERSITY ANALYSIS")
    print("=" * 70)
    diversity = {}
    for enc_name, gen_fn in encoders:
        H_ffts = gen_fn(NZ)
        diversity[enc_name] = analyze_psf_diversity(enc_name, H_ffts)

    results = {}

    # Config A: 3D only
    print("\n" + "=" * 70)
    print(f"CONFIG A: 3D DEPTH ONLY ({NZ}:1)")
    print("=" * 70)
    for enc_name, gen_fn in encoders:
        H_ffts = gen_fn(NZ)
        key = f"{enc_name}_3d_only"
        results[key] = run_3d_only(enc_name, H_ffts, NZ)

    # Config B: +Mask
    print("\n" + "=" * 70)
    print(f"CONFIG B: MASK + DEPTH ({NZ}:1)")
    print("=" * 70)
    for enc_name, gen_fn in encoders:
        H_ffts = gen_fn(NZ)
        key = f"{enc_name}_mask"
        results[key] = run_3d_mask(enc_name, H_ffts, NZ)

    # Config C: +M+W_lambda (4D spectral-depth)
    print("\n" + "=" * 70)
    print(f"CONFIG C: M + W_lambda + DEPTH ({NZ}×{NL}={NZ*NL}:1)")
    print("=" * 70)
    for enc_name, gen_fn in encoders:
        key = f"{enc_name}_m_wl"
        results[key] = run_4d_mask_spectral(enc_name, gen_fn, NZ, NL)

    # Config D: +M+W_t (4D temporal-depth)
    print("\n" + "=" * 70)
    print(f"CONFIG D: M + W_t + DEPTH ({NZ}×{NT}={NZ*NT}:1)")
    print("=" * 70)
    for enc_name, gen_fn in encoders:
        key = f"{enc_name}_m_wt"
        results[key] = run_4d_mask_temporal(enc_name, gen_fn, NZ, NT)

    # Config E: +M+W_lambda+W_t (5D full)
    print("\n" + "=" * 70)
    print(f"CONFIG E: M + W_lambda + W_t + DEPTH ({NZ}×{NL}×{NT}={NZ*NL*NT}:1)")
    print("=" * 70)
    for enc_name, gen_fn in encoders:
        key = f"{enc_name}_m_wl_wt"
        results[key] = run_5d_full(enc_name, gen_fn, NZ, NL, NT)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Best PSNR (dB) per configuration")
    print("=" * 70)
    configs = ["3d_only", "mask", "m_wl", "m_wt", "m_wl_wt"]
    config_labels = ["3D only", "+Mask", "+M+W_λ", "+M+W_t", "+M+W_λ+W_t"]
    comp_labels = [f"{NZ}:1", f"{NZ}:1", f"{NZ*NL}:1", f"{NZ*NT}:1",
                   f"{NZ*NL*NT}:1"]

    header = f"{'Encoder':<15}"
    for lbl, comp in zip(config_labels, comp_labels):
        header += f" {lbl} ({comp}):>14"
    # Print formatted
    print(f"\n{'Encoder':<15}", end="")
    for lbl, comp in zip(config_labels, comp_labels):
        print(f"  {lbl:>12}", end="")
    print()
    print(f"{'':15}", end="")
    for comp in comp_labels:
        print(f"  {'(' + comp + ')':>12}", end="")
    print()
    print("-" * 80)

    enc_names = ["Diffuser", "Multi-lens", "Meta-surface"]
    for enc in enc_names:
        print(f"{enc:<15}", end="")
        for cfg in configs:
            key = f"{enc}_{cfg}"
            if key in results:
                print(f"  {results[key]['best_psnr']:>10.1f}dB", end="")
            else:
                print(f"  {'N/A':>12}", end="")
        print()

    # Find best encoder per config
    print("\n" + "-" * 80)
    print(f"{'BEST':15}", end="")
    for cfg, lbl in zip(configs, config_labels):
        best_enc = ""
        best_val = -np.inf
        for enc in enc_names:
            key = f"{enc}_{cfg}"
            if key in results and results[key]["best_psnr"] > best_val:
                best_val = results[key]["best_psnr"]
                best_enc = enc
        print(f"  {best_enc[:10]:>12}", end="")
    print()

    # Save results
    output = {
        "diversity": diversity,
        "results": results,
        "config": {
            "NZ": NZ, "NL": NL, "NT": NT, "N": N,
            "N_SAMPLES": N_SAMPLES,
        }
    }
    out_path = "results/encoder_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
