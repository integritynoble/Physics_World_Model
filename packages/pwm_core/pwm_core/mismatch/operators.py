"""Operator correction for various imaging modalities.

This module provides functions to simulate mismatch in forward operators
and calibrate parameters to correct the mismatch.

Supported modalities:
- CT: Center of rotation calibration
- MRI: Coil sensitivity calibration
- CASSI: Dispersion step calibration
- CACTI: Mask timing calibration
- SPC: Gain/bias calibration
- Lensless: PSF shift calibration
- Ptychography: Position offset calibration
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_psnr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute PSNR between two arrays."""
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    max_val = max(x.max(), y.max(), 1.0)
    return float(10 * np.log10(max_val ** 2 / mse))


# =============================================================================
# CT: Center of Rotation
# =============================================================================

def ct_radon_forward(img: np.ndarray, angles: np.ndarray, cor_shift: int = 0) -> np.ndarray:
    """CT forward model (Radon transform) with center of rotation shift.

    Args:
        img: 2D image (n, n)
        angles: Array of projection angles in radians
        cor_shift: Center of rotation shift in pixels

    Returns:
        Sinogram (n_angles, n)
    """
    from scipy.ndimage import rotate
    n = img.shape[0]
    sinogram = np.zeros((len(angles), n), dtype=np.float32)
    for i, theta in enumerate(angles):
        rotated = rotate(img, np.degrees(theta), reshape=False, order=1)
        proj = rotated.sum(axis=0)
        if cor_shift != 0:
            proj = np.roll(proj, cor_shift)
        sinogram[i, :] = proj
    return sinogram


def ct_sart_tv_recon(
    sinogram: np.ndarray,
    angles: np.ndarray,
    n: int,
    cor_shift: int = 0,
    iters: int = 25,
    tv_weight: float = 0.08,
) -> np.ndarray:
    """SART-TV reconstruction for CT with center of rotation correction.

    Args:
        sinogram: Sinogram (n_angles, n)
        angles: Projection angles in radians
        n: Image size
        cor_shift: Center of rotation shift to correct
        iters: Number of iterations
        tv_weight: TV regularization weight

    Returns:
        Reconstructed image (n, n)
    """
    from scipy.ndimage import rotate
    try:
        from skimage.restoration import denoise_tv_chambolle
    except ImportError:
        denoise_tv_chambolle = None

    # Correct sinogram for center shift
    sino_corrected = np.zeros_like(sinogram)
    for i in range(len(angles)):
        sino_corrected[i] = np.roll(sinogram[i], -cor_shift)

    recon = np.zeros((n, n), dtype=np.float32)

    for _ in range(iters):
        for i, theta in enumerate(angles):
            rotated = rotate(recon, np.degrees(theta), reshape=False, order=1)
            proj = rotated.sum(axis=0)
            diff = (sino_corrected[i] - proj) / n
            back = np.tile(diff, (n, 1))
            back = rotate(back, -np.degrees(theta), reshape=False, order=1)
            recon += 0.2 * back

        if denoise_tv_chambolle is not None:
            recon = denoise_tv_chambolle(recon, weight=tv_weight, max_num_iter=5)

        recon = np.clip(recon, 0, 1)

    return recon.astype(np.float32)


def ct_calibrate_cor(
    sinogram: np.ndarray,
    angles: np.ndarray,
    n: int,
    cor_range: Tuple[int, int] = (-6, 7),
    recon_iters: int = 10,
) -> int:
    """Calibrate center of rotation for CT.

    Args:
        sinogram: Sinogram
        angles: Projection angles
        n: Image size
        cor_range: Range of CoR values to search
        recon_iters: Iterations for test reconstructions

    Returns:
        Best center of rotation value
    """
    best_cor = 0
    best_residual = float('inf')

    for test_cor in range(cor_range[0], cor_range[1]):
        recon_test = ct_sart_tv_recon(sinogram, angles, n, cor_shift=test_cor, iters=recon_iters)
        sino_test = ct_radon_forward(recon_test, angles, cor_shift=test_cor)
        residual = np.sum((sinogram - sino_test)**2)
        if residual < best_residual:
            best_residual = residual
            best_cor = test_cor

    return best_cor


# =============================================================================
# MRI: Coil Sensitivities
# =============================================================================

def mri_generate_coil_sensitivities(n: int, n_coils: int, seed: int = None) -> np.ndarray:
    """Generate synthetic coil sensitivity maps.

    Args:
        n: Image size
        n_coils: Number of coils
        seed: Random seed

    Returns:
        Sensitivities array (n_coils, n, n)
    """
    if seed is not None:
        np.random.seed(seed)

    sensitivities = np.zeros((n_coils, n, n), dtype=np.complex64)
    for c in range(n_coils):
        angle = 2 * np.pi * c / n_coils
        cx = n // 2 + int(40 * np.cos(angle))
        cy = n // 2 + int(40 * np.sin(angle))
        yy, xx = np.ogrid[:n, :n]
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        magnitude = np.exp(-dist / 50)
        phase = np.random.rand() * 2 * np.pi
        sensitivities[c] = magnitude * np.exp(1j * phase)

    # Normalize
    sos = np.sqrt(np.sum(np.abs(sensitivities)**2, axis=0))
    sensitivities = sensitivities / (sos + 1e-8)
    return sensitivities


def mri_forward_sense(
    img: np.ndarray,
    sens: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """SENSE forward model for parallel MRI.

    Args:
        img: Image (n, n), complex
        sens: Coil sensitivities (n_coils, n, n)
        mask: k-space undersampling mask (n, n), bool

    Returns:
        k-space data (n_coils, n, n)
    """
    n_coils = sens.shape[0]
    n = img.shape[0]
    k_data = np.zeros((n_coils, n, n), dtype=np.complex64)
    for c in range(n_coils):
        coil_img = img * sens[c]
        k_data[c] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coil_img)))
        k_data[c] *= mask
    return k_data


def mri_sense_recon(
    k_data: np.ndarray,
    sens: np.ndarray,
    mask: np.ndarray,
    iters: int = 30,
) -> np.ndarray:
    """SENSE reconstruction for parallel MRI.

    Args:
        k_data: k-space data (n_coils, n, n)
        sens: Coil sensitivities (n_coils, n, n)
        mask: Undersampling mask (n, n)
        iters: Number of CG iterations

    Returns:
        Reconstructed image (n, n)
    """
    n_coils, n, _ = k_data.shape
    recon = np.zeros((n, n), dtype=np.complex64)

    for _ in range(iters):
        # Forward
        residual = np.zeros_like(k_data)
        for c in range(n_coils):
            coil_img = recon * sens[c]
            k_pred = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coil_img)))
            residual[c] = (k_data[c] - k_pred * mask) * mask

        # Adjoint
        update = np.zeros((n, n), dtype=np.complex64)
        for c in range(n_coils):
            coil_update = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(residual[c])))
            update += np.conj(sens[c]) * coil_update

        recon += 0.5 * update

    return recon


def mri_estimate_sensitivities_acs(k_data: np.ndarray, acs_lines: int = 24) -> np.ndarray:
    """Estimate coil sensitivities from ACS (auto-calibration signal).

    Args:
        k_data: k-space data (n_coils, n, n)
        acs_lines: Number of ACS lines in center

    Returns:
        Estimated sensitivities (n_coils, n, n)
    """
    from scipy.ndimage import gaussian_filter

    n_coils, n, _ = k_data.shape
    acs_start = n // 2 - acs_lines // 2
    acs_end = n // 2 + acs_lines // 2

    sens_est = np.zeros_like(k_data)

    for c in range(n_coils):
        k_acs = np.zeros((n, n), dtype=np.complex64)
        k_acs[acs_start:acs_end, :] = k_data[c, acs_start:acs_end, :]
        img_lr = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k_acs)))
        sens_est[c] = gaussian_filter(np.abs(img_lr), sigma=3) * np.exp(1j * np.angle(img_lr))

    # Normalize
    sos = np.sqrt(np.sum(np.abs(sens_est)**2, axis=0))
    sens_est = sens_est / (sos + 1e-8)

    return sens_est


# =============================================================================
# CASSI: Dispersion Step
# =============================================================================

def cassi_shift(x: np.ndarray, step: int = 1) -> np.ndarray:
    """Shift spectral bands for CASSI dispersion.

    Args:
        x: Hyperspectral cube (H, W, nC)
        step: Dispersion step (pixels per band)

    Returns:
        Shifted cube (H, W + (nC-1)*step, nC)
    """
    h, w, nc = x.shape
    out = np.zeros((h, w + (nc - 1) * step, nc), dtype=x.dtype)
    for c in range(nc):
        out[:, c * step:c * step + w, c] = x[:, :, c]
    return out


def cassi_shift_back(y: np.ndarray, step: int, nc: int, w: int = None) -> np.ndarray:
    """Shift back spectral bands."""
    if w is None:
        w = y.shape[1] - (nc - 1) * step
    h = y.shape[0]
    out = np.zeros((h, w, nc), dtype=y.dtype)
    for c in range(nc):
        out[:, :, c] = y[:, c * step:c * step + w, c]
    return out


def cassi_forward(x: np.ndarray, mask: np.ndarray, step: int = 1) -> np.ndarray:
    """CASSI forward model.

    Args:
        x: Hyperspectral cube (H, W, nC)
        mask: Coded aperture mask (H, W)
        step: Dispersion step

    Returns:
        2D measurement
    """
    Phi = np.tile(mask[:, :, np.newaxis], (1, 1, x.shape[2]))
    masked = x * Phi
    shifted = cassi_shift(masked, step)
    return np.sum(shifted, axis=2)


def cassi_adjoint(y: np.ndarray, mask: np.ndarray, step: int, nc: int) -> np.ndarray:
    """CASSI adjoint."""
    h, w = mask.shape
    Phi = np.tile(mask[:, :, np.newaxis], (1, 1, nc))
    y_ext = np.tile(y[:, :, np.newaxis], (1, 1, nc))
    x = cassi_shift_back(y_ext, step, nc, w)
    return x * Phi


def cassi_gap_denoise(
    y: np.ndarray,
    mask: np.ndarray,
    step: int = 1,
    max_iter: int = 100,
    lam: float = 0.3,
) -> np.ndarray:
    """GAP-denoise for CASSI reconstruction.

    Args:
        y: 2D measurement
        mask: Coded aperture mask
        step: Dispersion step
        max_iter: Number of iterations
        lam: TV regularization weight

    Returns:
        Reconstructed hyperspectral cube
    """
    try:
        from skimage.restoration import denoise_tv_chambolle
    except ImportError:
        denoise_tv_chambolle = None
    from scipy.ndimage import gaussian_filter

    h, w = mask.shape
    nc = 28  # Default number of bands

    Phi = np.tile(mask[:, :, np.newaxis], (1, 1, nc))
    Phi_shifted = cassi_shift(Phi, step)
    Phi_sum = np.sum(Phi_shifted, axis=2)
    Phi_sum[Phi_sum == 0] = 1

    # Handle size mismatch
    y_w = y.shape[1]
    phi_w = Phi_sum.shape[1]
    if y_w != phi_w:
        if y_w < phi_w:
            y_pad = np.zeros((y.shape[0], phi_w), dtype=y.dtype)
            y_pad[:, :y_w] = y
            y = y_pad
        else:
            y = y[:, :phi_w]

    x = cassi_adjoint(y, mask, step, nc)
    for c in range(nc):
        x[:, :, c] = x[:, :, c] / (np.mean(mask) + 0.01)

    y1 = y.copy()

    for it in range(max_iter):
        yb = cassi_forward(x, mask, step)
        min_w = min(y.shape[1], yb.shape[1])
        y1[:, :min_w] = y1[:, :min_w] + (y[:, :min_w] - yb[:, :min_w])
        residual = np.zeros_like(yb)
        residual[:, :min_w] = y1[:, :min_w] - yb[:, :min_w]

        residual_norm = residual / np.maximum(Phi_sum[:, :residual.shape[1]], 1)
        x = x + cassi_adjoint(residual_norm, mask, step, nc)

        if denoise_tv_chambolle is not None:
            x = denoise_tv_chambolle(x, weight=lam, max_num_iter=5, channel_axis=2)
        else:
            for c in range(nc):
                x[:, :, c] = gaussian_filter(x[:, :, c], sigma=0.5)

        x = np.clip(x, 0, 1)

    return x.astype(np.float32)


def cassi_calibrate_step(
    y: np.ndarray,
    mask: np.ndarray,
    step_range: List[int] = [1, 2, 3, 4],
    recon_iters: int = 50,
) -> int:
    """Calibrate dispersion step for CASSI.

    Args:
        y: Measurement
        mask: Coded aperture
        step_range: Steps to search
        recon_iters: Iterations per test

    Returns:
        Best dispersion step
    """
    best_step = step_range[0]
    best_residual = float('inf')

    for test_step in step_range:
        x_test = cassi_gap_denoise(y, mask, step=test_step, max_iter=recon_iters)
        y_pred = cassi_forward(x_test, mask, step=test_step)
        min_w = min(y.shape[1], y_pred.shape[1])
        residual = np.sum((y[:, :min_w] - y_pred[:, :min_w])**2)
        if residual < best_residual:
            best_residual = residual
            best_step = test_step

    return best_step


# =============================================================================
# CACTI: Mask Timing
# =============================================================================

def cacti_forward(video: np.ndarray, masks: np.ndarray, timing_offset: int = 0) -> np.ndarray:
    """CACTI forward model with timing offset.

    Args:
        video: Video frames (H, W, nF)
        masks: Temporal masks (H, W, nF)
        timing_offset: Timing offset

    Returns:
        2D snapshot measurement
    """
    h, w, nF = video.shape
    measurement = np.zeros((h, w), dtype=np.float32)
    for f in range(nF):
        mask_idx = (f + timing_offset) % nF
        measurement += video[:, :, f] * masks[:, :, mask_idx]
    return measurement


def cacti_gap_tv(
    y: np.ndarray,
    masks: np.ndarray,
    timing_offset: int = 0,
    iters: int = 100,
    lam: float = 0.1,
) -> np.ndarray:
    """GAP-TV reconstruction for CACTI.

    Args:
        y: 2D measurement
        masks: Temporal masks (H, W, nF)
        timing_offset: Timing offset to apply
        iters: Number of iterations
        lam: TV weight

    Returns:
        Reconstructed video
    """
    try:
        from skimage.restoration import denoise_tv_chambolle
    except ImportError:
        denoise_tv_chambolle = None
    from scipy.ndimage import gaussian_filter

    h, w = y.shape
    nF = masks.shape[2]

    # Reorder masks according to timing
    masks_aligned = np.zeros_like(masks)
    for f in range(nF):
        mask_idx = (f + timing_offset) % nF
        masks_aligned[:, :, f] = masks[:, :, mask_idx]

    mask_sum = np.sum(masks_aligned, axis=2)
    mask_sum[mask_sum == 0] = 1

    x = np.tile(y[:, :, np.newaxis], (1, 1, nF)) / nF
    y1 = y.copy()

    for it in range(iters):
        yb = np.sum(x * masks_aligned, axis=2)
        y1 = y1 + (y - yb)

        for f in range(nF):
            x[:, :, f] = x[:, :, f] + masks_aligned[:, :, f] * (y1 - yb) / mask_sum

        if denoise_tv_chambolle is not None:
            for f in range(nF):
                x[:, :, f] = denoise_tv_chambolle(x[:, :, f], weight=lam, max_num_iter=3)
        else:
            for f in range(nF):
                x[:, :, f] = gaussian_filter(x[:, :, f], sigma=0.3)

        x = np.clip(x, 0, 1)

    return x.astype(np.float32)


def cacti_calibrate_timing(
    y: np.ndarray,
    masks: np.ndarray,
    recon_iters: int = 50,
) -> int:
    """Calibrate mask timing for CACTI.

    Returns:
        Best timing offset
    """
    nF = masks.shape[2]
    best_timing = 0
    best_residual = float('inf')

    for test_timing in range(nF):
        x_test = cacti_gap_tv(y, masks, timing_offset=test_timing, iters=recon_iters)
        y_pred = cacti_forward(x_test, masks, timing_offset=test_timing)
        residual = np.sum((y - y_pred)**2)
        if residual < best_residual:
            best_residual = residual
            best_timing = test_timing

    return best_timing


# =============================================================================
# SPC: Gain/Bias
# =============================================================================

def spc_forward(x: np.ndarray, Phi: np.ndarray, gain: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """SPC forward model with gain and bias.

    Args:
        x: Image (flattened)
        Phi: Measurement matrix
        gain: Detector gain
        bias: Detector bias

    Returns:
        Measurements
    """
    return gain * (Phi @ x) + bias


def spc_lsq_recon(
    y: np.ndarray,
    Phi: np.ndarray,
    gain: float = 1.0,
    bias: float = 0.0,
    reg: float = 0.001,
) -> np.ndarray:
    """Regularized least-squares reconstruction for SPC.

    Args:
        y: Measurements
        Phi: Measurement matrix
        gain: Gain to correct for
        bias: Bias to correct for
        reg: Regularization parameter

    Returns:
        Reconstructed image (flattened)
    """
    y_corrected = (y - bias) / max(gain, 0.01)
    AtA = Phi.T @ Phi + reg * np.eye(Phi.shape[1])
    Aty = Phi.T @ y_corrected
    x = np.linalg.solve(AtA, Aty)
    return np.clip(x, 0, 1).astype(np.float32)


def spc_calibrate_gain_bias(
    y: np.ndarray,
    Phi: np.ndarray,
    y_cal: np.ndarray = None,
    Phi_cal: np.ndarray = None,
    x_cal: np.ndarray = None,
    gain_range: Tuple[float, float] = (0.3, 2.0),
    bias_range: Tuple[float, float] = (-0.3, 0.3),
    n_steps: int = 31,
    reg: float = 0.001,
) -> Tuple[float, float]:
    """Calibrate gain and bias for SPC.

    Two modes:
    1. With calibration data (y_cal, x_cal): Direct estimation via linear regression
    2. Without: Grid search using reconstruction quality metric

    Args:
        y: Measurements
        Phi: Measurement matrix
        y_cal: Calibration measurements (optional)
        Phi_cal: Calibration measurement matrix (optional, defaults to Phi)
        x_cal: Known calibration signal (optional)
        gain_range: Range for grid search
        bias_range: Range for grid search
        n_steps: Grid search steps
        reg: Regularization parameter

    Returns:
        (best_gain, best_bias)
    """
    # Mode 1: Use calibration data if available
    if y_cal is not None and x_cal is not None:
        Phi_c = Phi_cal if Phi_cal is not None else Phi
        z = Phi_c @ x_cal
        # Linear regression: y_cal = gain * z + bias
        A = np.column_stack([z, np.ones_like(z)])
        params, _, _, _ = np.linalg.lstsq(A, y_cal, rcond=None)
        return float(max(params[0], 0.01)), float(params[1])

    # Mode 2: Grid search with TV-based quality metric
    best_gain, best_bias = 1.0, 0.0
    best_score = -float('inf')

    for test_gain in np.linspace(gain_range[0], gain_range[1], n_steps):
        for test_bias in np.linspace(bias_range[0], bias_range[1], n_steps):
            # Reconstruct
            y_corrected = (y - test_bias) / max(test_gain, 0.01)
            AtA = Phi.T @ Phi + reg * np.eye(Phi.shape[1])
            Aty = Phi.T @ y_corrected
            x_test = np.linalg.solve(AtA, Aty)
            x_test = np.clip(x_test, 0, 1)

            # Quality metric: negative total variation (smoother = better)
            n = int(np.sqrt(len(x_test)))
            x_2d = x_test.reshape(n, n)
            tv = np.sum(np.abs(np.diff(x_2d, axis=0))) + np.sum(np.abs(np.diff(x_2d, axis=1)))

            # Also penalize extreme values
            range_penalty = np.sum(np.maximum(0, x_test - 1)) + np.sum(np.maximum(0, -x_test))

            score = -tv - 100 * range_penalty

            if score > best_score:
                best_score = score
                best_gain, best_bias = test_gain, test_bias

    return best_gain, best_bias


# =============================================================================
# Lensless: PSF Shift
# =============================================================================

def lensless_forward(img: np.ndarray, psf: np.ndarray, shift: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Lensless forward model with PSF shift.

    Args:
        img: Image
        psf: Point spread function
        shift: PSF shift (dy, dx)

    Returns:
        Measurement
    """
    from scipy.signal import fftconvolve
    from scipy.ndimage import shift as ndshift

    psf_shifted = ndshift(psf, shift, order=1)
    return fftconvolve(img, psf_shifted, mode='same')


def lensless_admm_tv(
    y: np.ndarray,
    psf: np.ndarray,
    shift: Tuple[int, int] = (0, 0),
    iters: int = 50,
    rho: float = 1.0,
    tv_weight: float = 0.1,
) -> np.ndarray:
    """ADMM-TV reconstruction for lensless imaging.

    Args:
        y: Measurement
        psf: PSF
        shift: PSF shift to correct
        iters: Iterations
        rho: ADMM penalty parameter
        tv_weight: TV weight

    Returns:
        Reconstructed image
    """
    try:
        from skimage.restoration import denoise_tv_chambolle
    except ImportError:
        denoise_tv_chambolle = None
    from scipy.ndimage import shift as ndshift, gaussian_filter

    psf_shifted = ndshift(psf, shift, order=1)

    # FFT-based deconvolution with regularization
    psf_fft = np.fft.fft2(psf_shifted)
    psf_conj = np.conj(psf_fft)
    y_fft = np.fft.fft2(y)

    x = np.real(np.fft.ifft2(y_fft))
    z = x.copy()
    u = np.zeros_like(x)

    for _ in range(iters):
        # x-update (Wiener filter)
        x_fft = (psf_conj * y_fft + rho * np.fft.fft2(z - u)) / (np.abs(psf_fft)**2 + rho + 1e-6)
        x = np.real(np.fft.ifft2(x_fft))

        # z-update (TV denoising)
        if denoise_tv_chambolle is not None:
            z = denoise_tv_chambolle(x + u, weight=tv_weight/rho, max_num_iter=5)
        else:
            z = gaussian_filter(x + u, sigma=0.5)

        # u-update
        u = u + x - z

    return np.clip(z, 0, 1).astype(np.float32)


def lensless_calibrate_shift(
    y: np.ndarray,
    psf: np.ndarray,
    shift_range: Tuple[int, int] = (-4, 5),
    recon_iters: int = 20,
) -> Tuple[int, int]:
    """Calibrate PSF shift for lensless imaging.

    Returns:
        (best_shift_y, best_shift_x)
    """
    best_shift = (0, 0)
    best_residual = float('inf')

    for sx in range(shift_range[0], shift_range[1]):
        for sy in range(shift_range[0], shift_range[1]):
            x_test = lensless_admm_tv(y, psf, shift=(sy, sx), iters=recon_iters)
            y_pred = lensless_forward(x_test, psf, shift=(sy, sx))
            residual = np.sum((y - y_pred)**2)
            if residual < best_residual:
                best_residual = residual
                best_shift = (sy, sx)

    return best_shift


# =============================================================================
# Ptychography: Position Offset
# =============================================================================

def ptycho_get_positions(
    n: int,
    probe_size: int,
    step: int,
    offset: Tuple[int, int] = (0, 0),
) -> List[Tuple[int, int]]:
    """Generate scan positions with offset.

    Args:
        n: Object size
        probe_size: Probe size
        step: Step size
        offset: Position offset (y, x)

    Returns:
        List of (py, px) positions
    """
    positions = []
    for py in range(0, n - probe_size + 1, step):
        for px in range(0, n - probe_size + 1, step):
            positions.append((py + offset[0], px + offset[1]))
    return positions


def ptycho_forward(
    obj: np.ndarray,
    probe: np.ndarray,
    positions: List[Tuple[int, int]],
) -> np.ndarray:
    """Ptychography forward model.

    Args:
        obj: Complex object
        probe: Complex probe
        positions: List of scan positions

    Returns:
        Diffraction pattern intensities
    """
    probe_size = probe.shape[0]
    n = obj.shape[0]
    intensities = []

    for py, px in positions:
        py, px = int(round(py)), int(round(px))
        py = max(0, min(py, n - probe_size))
        px = max(0, min(px, n - probe_size))
        exit_wave = obj[py:py+probe_size, px:px+probe_size] * probe
        diffraction = np.abs(np.fft.fft2(exit_wave))**2
        intensities.append(diffraction)

    return np.array(intensities)


def ptycho_epie_recon(
    intensities: np.ndarray,
    probe: np.ndarray,
    positions: List[Tuple[int, int]],
    n: int,
    n_iters: int = 200,
) -> np.ndarray:
    """Extended PIE reconstruction.

    Args:
        intensities: Diffraction patterns
        probe: Initial probe estimate
        positions: Scan positions
        n: Object size
        n_iters: Number of iterations

    Returns:
        Reconstructed object amplitude
    """
    probe_size = probe.shape[0]
    obj = np.ones((n, n), dtype=np.complex64) * 0.6
    probe_est = probe.copy()

    for _ in range(n_iters):
        for idx, (py, px) in enumerate(positions):
            py, px = int(round(py)), int(round(px))
            py = max(0, min(py, n - probe_size))
            px = max(0, min(px, n - probe_size))

            exit_wave = obj[py:py+probe_size, px:px+probe_size] * probe_est
            exit_fft = np.fft.fft2(exit_wave)
            measured_amp = np.sqrt(np.maximum(intensities[idx], 0) + 1e-8)
            exit_fft_corrected = np.fft.ifftshift(measured_amp) * np.exp(1j * np.angle(exit_fft))
            exit_wave_new = np.fft.ifft2(exit_fft_corrected)

            # Object update
            obj_patch = obj[py:py+probe_size, px:px+probe_size]
            update_obj = np.conj(probe_est) / ((np.abs(probe_est)**2).max() + 1e-8)
            obj[py:py+probe_size, px:px+probe_size] += 0.9 * update_obj * (exit_wave_new - exit_wave)

            # Probe update
            update_probe = np.conj(obj_patch) / ((np.abs(obj_patch)**2).max() + 1e-8)
            probe_est += 0.5 * update_probe * (exit_wave_new - exit_wave)

    return np.abs(obj)


def ptycho_calibrate_offset(
    intensities: np.ndarray,
    probe: np.ndarray,
    n: int,
    probe_size: int,
    step: int,
    offset_range: Tuple[int, int] = (-6, 7),
    offset_step: int = 2,
    recon_iters: int = 80,
    gt_amplitude: np.ndarray = None,
) -> Tuple[int, int]:
    """Calibrate position offset for ptychography.

    Args:
        intensities: Diffraction patterns
        probe: Probe
        n: Object size
        probe_size: Probe size
        step: Scan step
        offset_range: Range to search
        offset_step: Search step
        recon_iters: Iterations per test
        gt_amplitude: Ground truth for PSNR comparison (optional)

    Returns:
        (best_offset_y, best_offset_x)
    """
    best_offset = (0, 0)
    best_metric = -float('inf')

    for ox in range(offset_range[0], offset_range[1], offset_step):
        for oy in range(offset_range[0], offset_range[1], offset_step):
            positions_test = ptycho_get_positions(n, probe_size, step, (oy, ox))
            recon_test = ptycho_epie_recon(intensities, probe, positions_test, n, n_iters=recon_iters)

            if gt_amplitude is not None:
                metric = compute_psnr(recon_test, gt_amplitude)
            else:
                # Use sharpness as metric
                metric = np.std(recon_test)

            if metric > best_metric:
                best_metric = metric
                best_offset = (oy, ox)

    return best_offset
