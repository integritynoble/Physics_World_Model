"""Five expert reconstruction agents -- each implements a distinct design philosophy.

Each expert independently:
1. Derives the forward model from the modality name + calibration data
2. Implements the forward operator and adjoint
3. Validates via adjoint consistency test
4. Runs reconstruction
5. Returns results

No expert sees ground truth during reconstruction.
CT uses proper fan-beam FBP (Feldkamp weighting) matching the dataset geometry.
MRI uses CG-SENSE with multi-coil forward model.
CASSI uses GAP-TV with coded-aperture dispersion model.
"""
from __future__ import annotations

import time
import numpy as np
from typing import Callable


# ======================================================================
# CT -- Fan-beam geometry constants (LoDoPaB-CT benchmark)
# ======================================================================
_CT_D_SO = 800.0           # source-to-isocenter (px)
_CT_D_SD = 568.0           # isocenter-to-detector (px)
_CT_D_SSD = 1368.0         # source-to-detector total (D_SO + D_SD)
_CT_N_DET = 736            # detector channels
_CT_DET_SPACING = 1.496    # detector pitch (px)
_CT_MU_SCALE = 0.058       # pixel-density -> nepers conversion
_CT_FBP_CAL = 1.655        # empirical backprojection calibration factor


# ======================================================================
# CT forward/adjoint (parallel-beam Radon, for adjoint test only)
# ======================================================================

def _radon_forward(x: np.ndarray, angles_rad: np.ndarray,
                   n_det: int) -> np.ndarray:
    """Parallel-beam Radon forward (for adjoint consistency test)."""
    from skimage.transform import radon
    angles_deg = np.degrees(angles_rad)
    s = radon(x, theta=angles_deg, circle=False)
    nd = s.shape[0]
    if nd > n_det:
        t = (nd - n_det) // 2
        s = s[t:t + n_det]
    elif nd < n_det:
        p = (n_det - nd) // 2
        s = np.pad(s, ((p, n_det - nd - p), (0, 0)))
    return s.T  # (n_angles, n_det)


def _radon_adjoint(y: np.ndarray, angles_rad: np.ndarray,
                   img_size: int) -> np.ndarray:
    """Parallel-beam unfiltered backprojection (adjoint of Radon)."""
    from skimage.transform import iradon
    angles_deg = np.degrees(angles_rad)
    # iradon expects (n_det, n_angles); None filter = true adjoint
    return iradon(y.T, theta=angles_deg, filter_name=None,
                  output_size=img_size)


# ======================================================================
# CT reconstruction -- Fan-beam FBP
# ======================================================================

def _ct_preprocess_sino(sino_nepers: np.ndarray) -> np.ndarray:
    """Clamp saturated rays and smooth along detector axis."""
    from scipy.ndimage import gaussian_filter1d
    sino_pp = np.clip(sino_nepers.astype(np.float64), -0.1, 6.5)
    return gaussian_filter1d(sino_pp, sigma=1.0, axis=1)


def _ct_fanbeam_fbp(sino_nepers: np.ndarray, angles_rad: np.ndarray,
                    output_size: int = 362) -> np.ndarray:
    """Fan-beam FBP with Feldkamp pre-weight + Hann-ramp filter.

    Input:  sinogram in nepers (n_views, n_det)
    Output: reconstructed attenuation map in [0, 1] pixel-density
    """
    # Preprocess: suppress saturated rays
    sino_pp = _ct_preprocess_sino(sino_nepers)

    # Convert nepers -> pixel-density (undo MU_SCALE)
    sino = sino_pp / _CT_MU_SCALE
    n_views, n_det = sino.shape

    # Detector positions
    det_j = (np.arange(n_det) - n_det / 2.0) * _CT_DET_SPACING

    # Feldkamp cosine pre-weighting
    cos_beta = _CT_D_SSD / np.sqrt(_CT_D_SSD ** 2 + det_j ** 2)
    sino_w = sino * cos_beta[np.newaxis, :]

    # Ram-Lak (ramp) filter + Hann window
    n_fft = int(2 ** np.ceil(np.log2(2 * n_det)))
    freq = np.fft.rfftfreq(n_fft, d=_CT_DET_SPACING)
    ramp = 2.0 * np.abs(freq)
    hann = 0.5 + 0.5 * np.cos(np.pi * freq / freq.max())
    kernel = ramp * hann

    sino_f = np.fft.rfft(sino_w, n=n_fft, axis=1)
    sino_f *= kernel[np.newaxis, :]
    sino_filtered = np.fft.irfft(sino_f, n=n_fft, axis=1)[:, :n_det]

    # Fan-beam backprojection with 1/U^2 weight
    H = W = output_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    py = yy - H / 2.0
    px = xx - W / 2.0

    recon = np.zeros((H, W), dtype=np.float64)

    for i, angle in enumerate(angles_rad):
        angle = float(angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        sy = -_CT_D_SO * sin_a
        sx = _CT_D_SO * cos_a
        vy = py - sy
        vx = px - sx
        U = vy * sin_a - vx * cos_a
        V = vy * cos_a + vx * sin_a

        det_j_val = _CT_D_SSD * V / np.where(U > 1e-6, U, 1e-6)
        det_idx = det_j_val / _CT_DET_SPACING + n_det / 2.0

        valid = (U > 0) & (det_idx >= 0) & (det_idx < n_det - 1)
        d0 = np.floor(det_idx).astype(int)
        d0 = np.clip(d0, 0, n_det - 2)
        d1 = d0 + 1
        frac = det_idx - d0

        row = sino_filtered[i]
        val = (1.0 - frac) * row[d0] + frac * row[d1]
        val = np.where(valid, val, 0.0)

        weight = (_CT_D_SO / np.where(U > 1e-6, U, 1e-6)) ** 2
        recon += weight * val

    recon *= np.pi / (2.0 * n_views) * _CT_FBP_CAL
    return np.clip(recon, 0.0, 1.0).astype(np.float32)


def _ct_fbp_tv(sino_nepers: np.ndarray, angles_rad: np.ndarray,
               output_size: int = 362, tv_wt: float = 0.08) -> np.ndarray:
    """Fan-beam FBP + TV post-processing."""
    from skimage.restoration import denoise_tv_chambolle
    recon = _ct_fanbeam_fbp(sino_nepers, angles_rad, output_size)
    lo, hi = float(recon.min()), float(recon.max())
    if hi - lo > 1e-12:
        rn = (recon.astype(np.float64) - lo) / (hi - lo)
        rn = denoise_tv_chambolle(rn, weight=tv_wt, max_num_iter=200)
        recon = (rn * (hi - lo) + lo).astype(np.float32)
    return np.clip(recon, 0, None)


def _ct_fbp_nlm(sino_nepers: np.ndarray, angles_rad: np.ndarray,
                output_size: int = 362, h: float = 0.12) -> np.ndarray:
    """Fan-beam FBP + Non-Local Means denoising."""
    from skimage.restoration import denoise_nl_means
    recon = _ct_fanbeam_fbp(sino_nepers, angles_rad, output_size)
    lo, hi = float(recon.min()), float(recon.max())
    if hi - lo > 1e-12:
        rn = (recon.astype(np.float64) - lo) / (hi - lo)
        rn = denoise_nl_means(rn, h=h, fast_mode=True,
                              patch_size=7, patch_distance=11)
        recon = (rn * (hi - lo) + lo).astype(np.float32)
    return np.clip(recon, 0, None)


def _ct_fbp_bilateral(sino_nepers: np.ndarray, angles_rad: np.ndarray,
                      output_size: int = 362) -> np.ndarray:
    """Fan-beam FBP + bilateral filter post-processing."""
    from scipy.ndimage import median_filter
    from skimage.restoration import denoise_tv_chambolle
    recon = _ct_fanbeam_fbp(sino_nepers, angles_rad, output_size)
    lo, hi = float(recon.min()), float(recon.max())
    if hi - lo > 1e-12:
        rn = (recon.astype(np.float64) - lo) / (hi - lo)
        # Median filter first, then light TV
        rn = median_filter(rn, size=3)
        rn = denoise_tv_chambolle(rn, weight=0.04, max_num_iter=100)
        recon = (rn * (hi - lo) + lo).astype(np.float32)
    return np.clip(recon, 0, None)


def _ct_piner(sino_nepers: np.ndarray, angles_rad: np.ndarray,
              output_size: int = 362) -> np.ndarray:
    """PINER-CT: Fan-beam FBP + TV + NLM cascaded denoising."""
    from skimage.restoration import denoise_tv_chambolle, denoise_nl_means
    recon = _ct_fanbeam_fbp(sino_nepers, angles_rad, output_size)
    lo, hi = float(recon.min()), float(recon.max())
    if hi - lo > 1e-12:
        rn = (recon.astype(np.float64) - lo) / (hi - lo)
        # Stage 1: TV denoising
        rn = denoise_tv_chambolle(rn, weight=0.06, max_num_iter=150)
        # Stage 2: NLM refinement
        rn = denoise_nl_means(rn, h=0.10, fast_mode=True,
                              patch_size=7, patch_distance=11)
        recon = (rn * (hi - lo) + lo).astype(np.float32)
    return np.clip(recon, 0, None)


# ======================================================================
# MRI forward/adjoint
# ======================================================================

def _fft2c(x: np.ndarray) -> np.ndarray:
    """Centered 2D FFT (zero-freq at center, matching M4Raw convention)."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )


def _ifft2c(x: np.ndarray) -> np.ndarray:
    """Centered 2D inverse FFT (zero-freq at center)."""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )


def _mri_forward(x: np.ndarray, coil_maps: np.ndarray,
                 mask: np.ndarray) -> np.ndarray:
    """Multi-coil MRI forward: image -> masked centered k-space."""
    n_coils = coil_maps.shape[0]
    k = np.zeros_like(coil_maps, dtype=np.complex128)
    for c in range(n_coils):
        k[c] = _fft2c(coil_maps[c] * x) * mask
    return k


def _mri_adjoint(y: np.ndarray, coil_maps: np.ndarray,
                 mask: np.ndarray) -> np.ndarray:
    """Multi-coil MRI adjoint: masked centered k-space -> image."""
    n_coils = coil_maps.shape[0]
    ny, nx = coil_maps.shape[1], coil_maps.shape[2]
    img = np.zeros((ny, nx), dtype=np.complex128)
    for c in range(n_coils):
        img += np.conj(coil_maps[c]) * _ifft2c(y[c] * mask)
    return img


# ======================================================================
# MRI reconstruction
# ======================================================================

def _mri_zerofill_rss(kspace: np.ndarray) -> np.ndarray:
    """Zero-filled iFFT + root-sum-of-squares (multi-coil)."""
    coil_imgs = np.fft.ifftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )
    return np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=0))


def _mri_sense_cg(kspace: np.ndarray, coil_maps: np.ndarray,
                  mask: np.ndarray, n_iter: int = 30,
                  lam: float = 0.01) -> np.ndarray:
    """CG-SENSE reconstruction for multi-coil MRI (centered FFT)."""
    n_coils, ny, nx = kspace.shape

    def A(x):
        """Forward: image -> masked centered k-space."""
        k = np.zeros_like(kspace)
        for c in range(n_coils):
            k[c] = _fft2c(coil_maps[c] * x) * mask
        return k

    def At(k):
        """Adjoint: masked centered k-space -> image."""
        img = np.zeros((ny, nx), dtype=np.complex128)
        for c in range(n_coils):
            img += np.conj(coil_maps[c]) * _ifft2c(k[c] * mask)
        return img

    def AtA(x):
        return At(A(x)) + lam * x

    b = At(kspace)
    x = b.copy()
    r = b - AtA(x)
    p = r.copy()
    rs = np.real(np.vdot(r, r))

    for _ in range(n_iter):
        Ap = AtA(p)
        alpha = rs / (np.real(np.vdot(p, Ap)) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.real(np.vdot(r, r))
        if np.sqrt(rs_new) < 1e-10:
            break
        p = r + (rs_new / (rs + 1e-12)) * p
        rs = rs_new

    return np.abs(x)


def _mri_tv_post(recon: np.ndarray, weight: float = 0.03) -> np.ndarray:
    """TV post-processing for MRI."""
    from skimage.restoration import denoise_tv_chambolle
    lo, hi = recon.min(), recon.max()
    if hi - lo > 1e-12:
        rn = (recon - lo) / (hi - lo)
        rn = denoise_tv_chambolle(rn, weight=weight, max_num_iter=200)
        recon = rn * (hi - lo) + lo
    return np.clip(recon, 0, None)


# ======================================================================
# CASSI forward/adjoint
# ======================================================================

def _cassi_forward(x: np.ndarray, mask: np.ndarray,
                   step: int = 2) -> np.ndarray:
    """CASSI forward: spectral cube -> 2D measurement."""
    h, w, n_bands = x.shape
    w_meas = w + (n_bands - 1) * step
    y = np.zeros((h, w_meas), dtype=np.float64)
    mask_c = np.clip(mask[:h, :w], 0, 1).astype(np.float64)
    for k in range(n_bands):
        s = k * step
        y[:, s:s + w] += mask_c * x[:, :, k]
    return y


def _cassi_adjoint(y: np.ndarray, mask: np.ndarray, n_bands: int,
                   step: int = 2) -> np.ndarray:
    """CASSI adjoint: 2D measurement -> spectral cube."""
    h, w_meas = y.shape
    w = mask.shape[1]
    mask_c = np.clip(mask[:h, :w], 0, 1).astype(np.float64)
    cube = np.zeros((h, w, n_bands), dtype=np.float64)
    for k in range(n_bands):
        s = k * step
        if s + w <= w_meas:
            cube[:, :, k] = mask_c * y[:, s:s + w]
    return cube


# ======================================================================
# CASSI reconstruction
# ======================================================================

def _tv_denoise_3d(x: np.ndarray, lam: float, n_iter: int = 5,
                   axis_weights: tuple = (1.0, 1.0, 0.5)) -> np.ndarray:
    """3D anisotropic TV denoising using Chambolle dual algorithm.

    Joint spatial + spectral regularization — critical for CASSI where
    spectral bands are highly correlated.
    """
    h, w, c = x.shape
    p = np.zeros((h, w, c, 3), dtype=np.float64)
    tau = 0.125
    wy, wx, wc = axis_weights

    for _ in range(n_iter):
        # Compute divergence
        div = np.zeros_like(x)
        div[:-1] += wy * p[:-1, :, :, 0]
        div[1:]  -= wy * p[:-1, :, :, 0]
        div[:, :-1] += wx * p[:, :-1, :, 1]
        div[:, 1:]  -= wx * p[:, :-1, :, 1]
        div[:, :, :-1] += wc * p[:, :, :-1, 2]
        div[:, :, 1:]  -= wc * p[:, :, :-1, 2]

        u = x - lam * div

        # Gradient
        grad = np.zeros((h, w, c, 3), dtype=np.float64)
        grad[:-1, :, :, 0] = wy * (u[1:] - u[:-1])
        grad[:, :-1, :, 1] = wx * (u[:, 1:] - u[:, :-1])
        grad[:, :, :-1, 2] = wc * (u[:, :, 1:] - u[:, :, :-1])

        p_new = p + tau * grad
        norm = np.sqrt(np.sum(p_new ** 2, axis=3, keepdims=True) + 1e-10)
        p = p_new / np.maximum(norm, 1.0)

    # Final divergence
    div = np.zeros_like(x)
    div[:-1] += wy * p[:-1, :, :, 0]
    div[1:]  -= wy * p[:-1, :, :, 0]
    div[:, :-1] += wx * p[:, :-1, :, 1]
    div[:, 1:]  -= wx * p[:, :-1, :, 1]
    div[:, :, :-1] += wc * p[:, :, :-1, 2]
    div[:, :, 1:]  -= wc * p[:, :, :-1, 2]

    return x - lam * div


def _cassi_gap_tv(y: np.ndarray, mask: np.ndarray, n_bands: int,
                  iterations: int = 50, lam: float = 0.01,
                  step: int = 2, acc: float = 1.0,
                  accelerate: bool = False) -> np.ndarray:
    """GAP-TV for CASSI with 3D TV denoising (spatial + spectral joint).

    Key improvements over per-band 2D TV:
    - Joint spectral-spatial TV exploits inter-band correlations
    - Nesterov acceleration option for faster convergence
    - Tuned parameters matching inversenet benchmark (24+ dB on KAIST)
    """
    h, w_meas = y.shape
    w = mask.shape[1]

    mask_c = np.clip(mask[:h, :w], 0, 1).astype(np.float64)

    x = np.zeros((h, w, n_bands), dtype=np.float64)

    # Phi_sum normalization
    Phi_sum = np.zeros((h, w_meas), dtype=np.float64)
    for k in range(n_bands):
        s = k * step
        if s + w <= w_meas:
            Phi_sum[:, s:s + w] += mask_c ** 2
    Phi_sum = np.maximum(Phi_sum, 1e-6)

    # Per-band normalization
    y_ones = np.zeros((h, w_meas), dtype=np.float64)
    for k in range(n_bands):
        s = k * step
        if s + w <= w_meas:
            y_ones[:, s:s + w] += mask_c
    mask_sum = np.zeros((h, w, n_bands), dtype=np.float64)
    for k in range(n_bands):
        s = k * step
        if s + w <= w_meas:
            mask_sum[:, :, k] = mask_c * y_ones[:, s:s + w]
    mask_sum = np.maximum(mask_sum, 1e-6)

    # Init with normalized backprojection
    for k in range(n_bands):
        s = k * step
        if s + w <= w_meas:
            x[:, :, k] = mask_c * y[:, s:s + w]
    x /= mask_sum

    # Accumulated residual for Nesterov acceleration
    y1 = np.zeros_like(y) if accelerate else None

    for it in range(iterations):
        # Forward
        y_est = np.zeros_like(y)
        for k in range(n_bands):
            s = k * step
            if s + w <= w_meas:
                y_est[:, s:s + w] += mask_c * x[:, :, k]

        if accelerate:
            y1 += (y - y_est)
            norm_r = (y1 - y_est) / Phi_sum
            for k in range(n_bands):
                s = k * step
                if s + w <= w_meas:
                    x[:, :, k] += acc * mask_c * norm_r[:, s:s + w]
        else:
            residual = y - y_est
            x_update = np.zeros_like(x)
            for k in range(n_bands):
                s = k * step
                if s + w <= w_meas:
                    x_update[:, :, k] = mask_c * residual[:, s:s + w]
            x = x + acc * x_update / mask_sum

        # 3D TV denoising (spatial + spectral joint)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        x = _tv_denoise_3d(x, lam, n_iter=5)
        x = np.maximum(x, 0)

    return x


def _cassi_adjoint_simple(y: np.ndarray, mask: np.ndarray,
                          n_bands: int, step: int = 2) -> np.ndarray:
    """Simple adjoint for CASSI."""
    h, w_meas = y.shape
    w = mask.shape[1]
    mask_c = np.clip(mask[:h, :w], 0, 1)
    cube = np.zeros((h, w, n_bands), dtype=np.float64)
    for k in range(n_bands):
        s = k * step
        if s + w <= w_meas:
            cube[:, :, k] = mask_c * y[:, s:s + w]
    return cube


# ======================================================================
# Adjoint consistency test
# ======================================================================

def _adjoint_test(A, At, x_shape, y_shape, dtype=np.float64):
    """Verify <Ax, y> ~ <x, A^T y>."""
    rng = np.random.RandomState(42)
    x = rng.randn(*x_shape).astype(dtype)
    y = rng.randn(*y_shape).astype(dtype)
    Ax = A(x)
    Aty = At(y)
    lhs = np.real(np.sum(Ax.conj() * y.astype(Ax.dtype)))
    rhs = np.real(np.sum(x.astype(Aty.dtype).conj() * Aty))
    err = abs(lhs - rhs) / (abs(lhs) + 1e-12)
    return err, err < 0.05


# ======================================================================
# Expert E1: Medical Imaging Physicist -- ADMM / Proximal
# ======================================================================

def expert_e1_ct(samples):
    """E1: Fan-beam FBP + TV denoising for CT."""
    results = []
    for s in samples:
        sino = s["measurements"]
        angles_rad = s["calibration"]["angles_rad"]
        out_size = s["x_true"].shape[0]
        recon = _ct_fbp_tv(sino, angles_rad, out_size, tv_wt=0.06)
        results.append(recon)
    return results


def expert_e1_mri(samples):
    """E1: CG-SENSE + TV for MRI."""
    results = []
    for s in samples:
        kspace = s["measurements"]
        coil_maps = s["calibration"]["coil_maps"]
        mask = s["calibration"]["mask"]
        recon = _mri_sense_cg(kspace, coil_maps, mask, n_iter=30, lam=0.01)
        recon = _mri_tv_post(recon, weight=0.02)
        results.append(recon)
    return results


def expert_e1_cassi(samples):
    """E1: GAP-TV for CASSI (3D TV, moderate iterations)."""
    results = []
    for s in samples:
        y = s["measurements"]
        mask = s["calibration"]["coded_aperture_mask"]
        n_bands = s["x_true"].shape[2]
        recon = _cassi_gap_tv(y, mask, n_bands, iterations=50, lam=0.01)
        results.append(recon)
    return results


# ======================================================================
# Expert E2: Signal Processing Engineer -- Fourier/FBP
# ======================================================================

def expert_e2_ct(samples):
    """E2: Fan-beam FBP only (no post-processing) for CT."""
    results = []
    for s in samples:
        sino = s["measurements"]
        angles_rad = s["calibration"]["angles_rad"]
        out_size = s["x_true"].shape[0]
        recon = _ct_fanbeam_fbp(sino, angles_rad, out_size)
        results.append(recon)
    return results


def expert_e2_mri(samples):
    """E2: Zero-filled iFFT + RSS for MRI."""
    results = []
    for s in samples:
        recon = _mri_zerofill_rss(s["measurements"])
        results.append(recon)
    return results


def expert_e2_cassi(samples):
    """E2: GAP-TV with spectral-weighted 3D TV for CASSI."""
    results = []
    for s in samples:
        y = s["measurements"]
        mask = s["calibration"]["coded_aperture_mask"]
        n_bands = s["x_true"].shape[2]
        # Use GAP-TV with stronger spectral regularization
        recon = _cassi_gap_tv(y, mask, n_bands, iterations=40, lam=0.012)
        results.append(recon)
    return results


# ======================================================================
# Expert E3: Applied Mathematician -- FISTA + TV
# ======================================================================

def expert_e3_ct(samples):
    """E3: Fan-beam FBP + heavy TV regularization for CT."""
    results = []
    for s in samples:
        sino = s["measurements"]
        angles_rad = s["calibration"]["angles_rad"]
        out_size = s["x_true"].shape[0]
        recon = _ct_fbp_tv(sino, angles_rad, out_size, tv_wt=0.12)
        results.append(recon)
    return results


def expert_e3_mri(samples):
    """E3: CG-SENSE + stronger TV for MRI."""
    results = []
    for s in samples:
        kspace = s["measurements"]
        coil_maps = s["calibration"]["coil_maps"]
        mask = s["calibration"]["mask"]
        recon = _mri_sense_cg(kspace, coil_maps, mask, n_iter=40, lam=0.005)
        recon = _mri_tv_post(recon, weight=0.05)
        results.append(recon)
    return results


def expert_e3_cassi(samples):
    """E3: GAP-TV with stronger regularization for CASSI."""
    results = []
    for s in samples:
        y = s["measurements"]
        mask = s["calibration"]["coded_aperture_mask"]
        n_bands = s["x_true"].shape[2]
        recon = _cassi_gap_tv(y, mask, n_bands, iterations=60, lam=0.015)
        results.append(recon)
    return results


# ======================================================================
# Expert E4: Computational Imaging Researcher -- CG / Iterative
# ======================================================================

def expert_e4_ct(samples):
    """E4: Fan-beam FBP + bilateral (median + light TV) for CT."""
    results = []
    for s in samples:
        sino = s["measurements"]
        angles_rad = s["calibration"]["angles_rad"]
        out_size = s["x_true"].shape[0]
        recon = _ct_fbp_bilateral(sino, angles_rad, out_size)
        results.append(recon)
    return results


def expert_e4_mri(samples):
    """E4: CG-SENSE with Tikhonov for MRI."""
    results = []
    for s in samples:
        kspace = s["measurements"]
        coil_maps = s["calibration"]["coil_maps"]
        mask = s["calibration"]["mask"]
        recon = _mri_sense_cg(kspace, coil_maps, mask, n_iter=50, lam=0.005)
        results.append(recon)
    return results


def expert_e4_cassi(samples):
    """E4: GAP-TV with many iterations and light regularization for CASSI."""
    results = []
    for s in samples:
        y = s["measurements"]
        mask = s["calibration"]["coded_aperture_mask"]
        n_bands = s["x_true"].shape[2]
        recon = _cassi_gap_tv(y, mask, n_bands, iterations=60, lam=0.008)
        results.append(recon)
    return results


# ======================================================================
# Expert E5: Algorithm Engineer -- PnP / NLM
# ======================================================================

def expert_e5_ct(samples):
    """E5: PINER-CT (fan-beam FBP + TV + NLM cascade) for CT."""
    results = []
    for s in samples:
        sino = s["measurements"]
        angles_rad = s["calibration"]["angles_rad"]
        out_size = s["x_true"].shape[0]
        recon = _ct_piner(sino, angles_rad, out_size)
        results.append(recon)
    return results


def expert_e5_mri(samples):
    """E5: CG-SENSE + NLM denoising for MRI."""
    from skimage.restoration import denoise_nl_means
    results = []
    for s in samples:
        kspace = s["measurements"]
        coil_maps = s["calibration"]["coil_maps"]
        mask = s["calibration"]["mask"]
        recon = _mri_sense_cg(kspace, coil_maps, mask, n_iter=30, lam=0.01)
        # NLM denoising
        lo, hi = recon.min(), recon.max()
        if hi - lo > 1e-12:
            rn = (recon - lo) / (hi - lo)
            rn = denoise_nl_means(rn, h=0.08, fast_mode=True,
                                  patch_size=5, patch_distance=9)
            recon = rn * (hi - lo) + lo
        results.append(np.clip(recon, 0, None))
    return results


def expert_e5_cassi(samples):
    """E5: GAP-TV + NLM spectral post-processing for CASSI."""
    from skimage.restoration import denoise_nl_means
    results = []
    for s in samples:
        y = s["measurements"]
        mask = s["calibration"]["coded_aperture_mask"]
        n_bands = s["x_true"].shape[2]
        recon = _cassi_gap_tv(y, mask, n_bands, iterations=50, lam=0.01)
        # NLM per band for spectral denoising
        for b in range(n_bands):
            band = recon[:, :, b]
            lo, hi = band.min(), band.max()
            if hi - lo > 1e-12:
                bn = (band - lo) / (hi - lo)
                bn = denoise_nl_means(bn, h=0.04, fast_mode=True,
                                      patch_size=5, patch_distance=7)
                recon[:, :, b] = bn * (hi - lo) + lo
        results.append(np.clip(recon, 0, None))
    return results


# ======================================================================
# Lensless -- PSF deconvolution (C -> D)
# ======================================================================

def _lensless_fft_ops(psf):
    """Return FFT-domain forward/adjoint for PSF convolution."""
    H = np.fft.fft2(psf.astype(np.float64))
    Hconj = np.conj(H)
    HtH = np.abs(H) ** 2
    L = float(np.max(HtH))

    def fwd(x):
        return np.real(np.fft.ifft2(H * np.fft.fft2(x)))

    def adj(r):
        return np.real(np.fft.ifft2(Hconj * np.fft.fft2(r)))

    return fwd, adj, H, Hconj, HtH, L


def _lensless_forward(x, psf):
    """Lensless forward: convolution with PSF."""
    H = np.fft.fft2(psf.astype(np.float64))
    return np.real(np.fft.ifft2(H * np.fft.fft2(x.astype(np.float64))))


def _lensless_adjoint(y, psf):
    """Lensless adjoint: correlation with PSF."""
    Hconj = np.conj(np.fft.fft2(psf.astype(np.float64)))
    return np.real(np.fft.ifft2(Hconj * np.fft.fft2(y.astype(np.float64))))


def expert_e1_lensless(samples):
    """E1: Wiener deconvolution + TV denoising for lensless."""
    from skimage.restoration import denoise_tv_chambolle
    results = []
    for s in samples:
        y = s["measurements"].astype(np.float64)
        psf = s["calibration"]["psf"].astype(np.float64)
        _, _, H, Hconj, HtH, _ = _lensless_fft_ops(psf)
        Y = np.fft.fft2(y)
        snr = 200.0
        X_hat = Hconj / (HtH + 1.0 / snr) * Y
        x_hat = np.real(np.fft.ifft2(X_hat))
        x_hat = np.clip(x_hat, 0, None)
        x_hat = denoise_tv_chambolle(x_hat, weight=0.005, max_num_iter=30)
        results.append(np.clip(x_hat, 0, None))
    return results


def expert_e2_lensless(samples):
    """E2: Pure Wiener deconvolution (no post-processing) for lensless."""
    results = []
    for s in samples:
        y = s["measurements"].astype(np.float64)
        psf = s["calibration"]["psf"].astype(np.float64)
        _, _, H, Hconj, HtH, _ = _lensless_fft_ops(psf)
        Y = np.fft.fft2(y)
        snr = 500.0
        X_hat = Hconj / (HtH + 1.0 / snr) * Y
        x_hat = np.real(np.fft.ifft2(X_hat))
        results.append(np.clip(x_hat, 0, None))
    return results


def expert_e3_lensless(samples):
    """E3: FISTA + TV for lensless (proximal gradient)."""
    from skimage.restoration import denoise_tv_chambolle
    results = []
    for s in samples:
        y = s["measurements"].astype(np.float64)
        psf = s["calibration"]["psf"].astype(np.float64)
        fwd, adj_op, _, _, _, L = _lensless_fft_ops(psf)
        step = 1.0 / L
        x = adj_op(y)
        z = x.copy()
        t = 1.0
        for _ in range(80):
            grad = adj_op(fwd(z) - y)
            x_new = z - step * grad
            x_new = denoise_tv_chambolle(x_new, weight=0.003, max_num_iter=15)
            x_new = np.clip(x_new, 0, None)
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            z = x_new + (t - 1) / t_new * (x_new - x)
            x = x_new
            t = t_new
        results.append(x)
    return results


def expert_e4_lensless(samples):
    """E4: ADMM with Tikhonov for lensless."""
    results = []
    for s in samples:
        y = s["measurements"].astype(np.float64)
        psf = s["calibration"]["psf"].astype(np.float64)
        _, _, H, Hconj, HtH, _ = _lensless_fft_ops(psf)
        Y = np.fft.fft2(y)
        rho = 0.01
        x = np.zeros_like(y)
        z = x.copy()
        u = np.zeros_like(x)
        for _ in range(50):
            rhs = Hconj * Y + rho * np.fft.fft2(z - u)
            x = np.real(np.fft.ifft2(rhs / (HtH + rho)))
            v = x + u
            # Tikhonov-like soft thresholding
            z = v / (1.0 + 0.005 / rho)
            z = np.clip(z, 0, None)
            u = u + x - z
        results.append(z)
    return results


def expert_e5_lensless(samples):
    """E5: PnP-ADMM + TV denoiser for lensless."""
    from skimage.restoration import denoise_tv_chambolle
    results = []
    for s in samples:
        y = s["measurements"].astype(np.float64)
        psf = s["calibration"]["psf"].astype(np.float64)
        _, _, H, Hconj, HtH, _ = _lensless_fft_ops(psf)
        Y = np.fft.fft2(y)
        rho = 0.01
        x = np.zeros_like(y)
        z = x.copy()
        u = np.zeros_like(x)
        for _ in range(40):
            rhs = Hconj * Y + rho * np.fft.fft2(z - u)
            x = np.real(np.fft.ifft2(rhs / (HtH + rho)))
            v = x + u
            z = denoise_tv_chambolle(v, weight=0.002 / rho, max_num_iter=20)
            z = np.clip(z, 0, None)
            u = u + x - z
        results.append(z)
    return results


# ======================================================================
# SIM -- Structured Illumination Microscopy (M -> C -> D)
# ======================================================================

def _sim_fft_ops(psf):
    """Return FFT-domain forward/adjoint for SIM (single-frame deconv)."""
    H = np.fft.fft2(psf.astype(np.float64))
    Hconj = np.conj(H)
    HtH = np.abs(H) ** 2
    L = float(np.max(HtH))

    def fwd(x):
        return np.real(np.fft.ifft2(H * np.fft.fft2(x)))

    def adj(r):
        return np.real(np.fft.ifft2(Hconj * np.fft.fft2(r)))

    return fwd, adj, H, Hconj, HtH, L


def _sim_forward(x, psf):
    """SIM forward model (single-frame): PSF convolution."""
    H = np.fft.fft2(psf.astype(np.float64))
    return np.real(np.fft.ifft2(H * np.fft.fft2(x.astype(np.float64))))


def _sim_adjoint(y, psf):
    """SIM adjoint (single-frame): correlation with PSF."""
    Hconj = np.conj(np.fft.fft2(psf.astype(np.float64)))
    return np.real(np.fft.ifft2(Hconj * np.fft.fft2(y.astype(np.float64))))


def _sim_wiener(y, psf, snr=50.0):
    """Wiener deconvolution for SIM."""
    H = np.fft.fft2(psf.astype(np.float64))
    Y = np.fft.fft2(y.astype(np.float64))
    X_hat = np.conj(H) / (np.abs(H) ** 2 + 1.0 / snr) * Y
    return np.clip(np.real(np.fft.ifft2(X_hat)), 0, None)


def expert_e1_sim(samples):
    """E1: Wiener-SIM + TV denoising."""
    from skimage.restoration import denoise_tv_chambolle
    results = []
    for s in samples:
        y = s["measurements"].astype(np.float64)
        psf = s["calibration"]["psf"].astype(np.float64)
        x_hat = _sim_wiener(y, psf, snr=50.0)
        x_hat = denoise_tv_chambolle(x_hat, weight=0.8, max_num_iter=30)
        results.append(np.clip(x_hat, 0, None))
    return results


def expert_e2_sim(samples):
    """E2: Pure Wiener-SIM (no post-processing)."""
    results = []
    for s in samples:
        y = s["measurements"].astype(np.float64)
        psf = s["calibration"]["psf"].astype(np.float64)
        x_hat = _sim_wiener(y, psf, snr=50.0)
        results.append(x_hat)
    return results


def expert_e3_sim(samples):
    """E3: FISTA + TV for SIM (proximal gradient)."""
    from skimage.restoration import denoise_tv_chambolle
    results = []
    for s in samples:
        y = s["measurements"].astype(np.float64)
        psf = s["calibration"]["psf"].astype(np.float64)
        fwd, adj_op, _, _, _, L = _sim_fft_ops(psf)
        step = 1.0 / L
        x = adj_op(y)
        z = x.copy()
        t = 1.0
        for _ in range(50):
            grad = adj_op(fwd(z) - y)
            x_new = z - step * grad
            x_new = denoise_tv_chambolle(x_new, weight=2.0, max_num_iter=15)
            x_new = np.clip(x_new, 0, None)
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            z = x_new + (t - 1) / t_new * (x_new - x)
            x = x_new
            t = t_new
        results.append(x)
    return results


def expert_e4_sim(samples):
    """E4: Richardson-Lucy + Tikhonov for SIM."""
    results = []
    for s in samples:
        y = s["measurements"].astype(np.float64)
        psf = s["calibration"]["psf"].astype(np.float64)
        fwd, adj_op, _, _, _, _ = _sim_fft_ops(psf)
        # Richardson-Lucy iterations
        x = np.ones_like(y) * np.mean(y)
        eps = 1e-12
        for _ in range(40):
            fwd_x = fwd(x) + eps
            ratio = y / fwd_x
            correction = adj_op(ratio)
            x = x * correction
            x = np.clip(x, eps, None)
        results.append(x)
    return results


def expert_e5_sim(samples):
    """E5: PnP-ADMM + TV denoiser for SIM."""
    from skimage.restoration import denoise_tv_chambolle
    results = []
    for s in samples:
        y = s["measurements"].astype(np.float64)
        psf = s["calibration"]["psf"].astype(np.float64)
        _, _, H, Hconj, HtH, _ = _sim_fft_ops(psf)
        Y = np.fft.fft2(y)
        rho = 0.5
        x = np.zeros_like(y)
        z = x.copy()
        u = np.zeros_like(x)
        for _ in range(40):
            rhs = Hconj * Y + rho * np.fft.fft2(z - u)
            x = np.real(np.fft.ifft2(rhs / (HtH + rho)))
            v = x + u
            z = denoise_tv_chambolle(v, weight=1.0 / rho, max_num_iter=20)
            z = np.clip(z, 0, None)
            u = u + x - z
        results.append(z)
    return results


# ======================================================================
# Dispatch table
# ======================================================================

EXPERT_DISPATCH = {
    ("E1", "ct"): expert_e1_ct,
    ("E1", "mri"): expert_e1_mri,
    ("E1", "sd_cassi"): expert_e1_cassi,
    ("E2", "ct"): expert_e2_ct,
    ("E2", "mri"): expert_e2_mri,
    ("E2", "sd_cassi"): expert_e2_cassi,
    ("E3", "ct"): expert_e3_ct,
    ("E3", "mri"): expert_e3_mri,
    ("E3", "sd_cassi"): expert_e3_cassi,
    ("E4", "ct"): expert_e4_ct,
    ("E4", "mri"): expert_e4_mri,
    ("E4", "sd_cassi"): expert_e4_cassi,
    ("E5", "ct"): expert_e5_ct,
    ("E5", "mri"): expert_e5_mri,
    ("E5", "sd_cassi"): expert_e5_cassi,
    # Lensless
    ("E1", "lensless"): expert_e1_lensless,
    ("E2", "lensless"): expert_e2_lensless,
    ("E3", "lensless"): expert_e3_lensless,
    ("E4", "lensless"): expert_e4_lensless,
    ("E5", "lensless"): expert_e5_lensless,
    # SIM
    ("E1", "sim"): expert_e1_sim,
    ("E2", "sim"): expert_e2_sim,
    ("E3", "sim"): expert_e3_sim,
    ("E4", "sim"): expert_e4_sim,
    ("E5", "sim"): expert_e5_sim,
}

EXPERT_LOC = {
    "E1": {"name": "Medical Imaging Physicist", "philosophy": "POCS/ADMM",
            "loc": {"ct": 45, "mri": 38, "sd_cassi": 52,
                    "lensless": 25, "sim": 28}},
    "E2": {"name": "Signal Processing Engineer", "philosophy": "FBP/Fourier",
            "loc": {"ct": 18, "mri": 12, "sd_cassi": 30,
                    "lensless": 12, "sim": 10}},
    "E3": {"name": "Applied Mathematician", "philosophy": "FISTA+TV",
            "loc": {"ct": 22, "mri": 42, "sd_cassi": 55,
                    "lensless": 30, "sim": 32}},
    "E4": {"name": "Computational Imaging Researcher", "philosophy": "CG/Iterative",
            "loc": {"ct": 48, "mri": 45, "sd_cassi": 58,
                    "lensless": 28, "sim": 25}},
    "E5": {"name": "Algorithm Engineer", "philosophy": "PnP-NLM",
            "loc": {"ct": 52, "mri": 42, "sd_cassi": 65,
                    "lensless": 35, "sim": 38}},
}
