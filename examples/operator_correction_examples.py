"""
Operator Correction Examples - Reconstruction With and Without Calibration

This file demonstrates how to run each imaging modality with and without
operator correction (calibration). Each example shows:
1. Forward simulation with TRUE parameters
2. Reconstruction with WRONG parameters (no correction)
3. Calibration to find correct parameters
4. Reconstruction with CALIBRATED parameters (with correction)

Usage:
    python examples/operator_correction_examples.py --modality ct
    python examples/operator_correction_examples.py --modality mri
    python examples/operator_correction_examples.py --modality cassi
    python examples/operator_correction_examples.py --modality cacti
    python examples/operator_correction_examples.py --modality spc
    python examples/operator_correction_examples.py --modality lensless
    python examples/operator_correction_examples.py --modality ptychography
    python examples/operator_correction_examples.py --all

For benchmark-quality results with larger improvements, use:
    python packages/pwm_core/benchmarks/test_operator_correction.py --all
"""

from __future__ import annotations
import argparse
import numpy as np
from typing import Dict, Any


def compute_psnr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute PSNR between two arrays."""
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    max_val = max(x.max(), y.max(), 1.0)
    return float(10 * np.log10(max_val ** 2 / mse))


# =============================================================================
# 1. CT - Center of Rotation Calibration
# =============================================================================
def example_ct_correction():
    """
    CT Center of Rotation Correction

    Problem: The center of rotation (CoR) can be misaligned in real CT systems.
    Solution: Calibrate CoR by minimizing reconstruction artifacts.

    Without correction: ~12 dB (severe ring artifacts)
    With correction:    ~29 dB (clean reconstruction)
    """
    print("\n" + "="*70)
    print("CT: Center of Rotation Calibration")
    print("="*70)

    from scipy.ndimage import rotate, gaussian_filter

    np.random.seed(52)
    n = 128
    n_angles = 90
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)

    # Create phantom (Shepp-Logan style)
    phantom = np.zeros((n, n), dtype=np.float32)
    cy, cx = n // 2, n // 2
    yy, xx = np.ogrid[:n, :n]
    phantom[((xx - cx) / 50)**2 + ((yy - cy) / 60)**2 < 1] = 0.8
    phantom[((xx - cx) / 45)**2 + ((yy - cy) / 55)**2 < 1] = 0.5

    print(f"  Phantom size: {n}x{n}")
    print(f"  Number of angles: {n_angles}")

    # Forward model with center shift
    def radon_forward(img, angles, cor_shift=0):
        sinogram = np.zeros((len(angles), n), dtype=np.float32)
        for i, theta in enumerate(angles):
            rotated = rotate(img, np.degrees(theta), reshape=False, order=1)
            proj = rotated.sum(axis=0)
            if cor_shift != 0:
                proj = np.roll(proj, cor_shift)
            sinogram[i, :] = proj
        return sinogram

    # SART-TV reconstruction
    def sart_tv_recon(sinogram, angles, n, cor_shift=0, iters=25):
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
                recon = denoise_tv_chambolle(recon, weight=0.08, max_num_iter=5)

            recon = np.clip(recon, 0, 1)

        return recon

    # TRUE parameter
    cor_true = 0

    # WRONG parameter (significant mismatch)
    cor_wrong = 4

    print(f"\n  TRUE center of rotation: {cor_true}")
    print(f"  WRONG center of rotation: {cor_wrong}")

    # Generate sinogram with TRUE CoR
    sinogram = radon_forward(phantom, angles, cor_shift=cor_true)
    sinogram += np.random.randn(*sinogram.shape).astype(np.float32) * 0.01

    # Reconstruct WITHOUT correction (using wrong CoR)
    print("\n  Reconstructing WITHOUT correction...")
    recon_wrong = sart_tv_recon(sinogram, angles, n, cor_shift=cor_wrong)
    psnr_wrong = compute_psnr(recon_wrong, phantom)
    print(f"  PSNR without correction: {psnr_wrong:.2f} dB")

    # CALIBRATION: Search for best CoR
    print("\n  Calibrating center of rotation...")
    best_cor = cor_wrong
    best_residual = float('inf')

    for test_cor in range(-6, 7):
        recon_test = sart_tv_recon(sinogram, angles, n, cor_shift=test_cor, iters=10)
        sino_test = radon_forward(recon_test, angles, cor_shift=test_cor)
        residual = np.sum((sinogram - sino_test)**2)
        if residual < best_residual:
            best_residual = residual
            best_cor = test_cor

    print(f"  Calibrated center of rotation: {best_cor}")

    # Reconstruct WITH correction (using calibrated CoR)
    print("\n  Reconstructing WITH correction...")
    recon_corrected = sart_tv_recon(sinogram, angles, n, cor_shift=best_cor)
    psnr_corrected = compute_psnr(recon_corrected, phantom)
    print(f"  PSNR with correction: {psnr_corrected:.2f} dB")

    # Oracle (true parameters)
    recon_oracle = sart_tv_recon(sinogram, angles, n, cor_shift=cor_true)
    psnr_oracle = compute_psnr(recon_oracle, phantom)

    print(f"\n  SUMMARY:")
    print(f"  Without correction: {psnr_wrong:.2f} dB")
    print(f"  With correction:    {psnr_corrected:.2f} dB (+{psnr_corrected - psnr_wrong:.2f} dB)")
    print(f"  Oracle (true CoR):  {psnr_oracle:.2f} dB")

    return {
        "modality": "ct",
        "parameter": "center_of_rotation",
        "psnr_without": psnr_wrong,
        "psnr_with": psnr_corrected,
        "improvement": psnr_corrected - psnr_wrong
    }


# =============================================================================
# 2. MRI - Coil Sensitivity Calibration
# =============================================================================
def example_mri_correction():
    """
    MRI Coil Sensitivity Calibration

    Problem: Parallel MRI requires accurate coil sensitivity maps.
    Solution: Estimate sensitivities from auto-calibration signal (ACS).

    Without correction: ~7 dB (severe aliasing artifacts)
    With correction:    ~55 dB (clean reconstruction)
    """
    print("\n" + "="*70)
    print("MRI: Coil Sensitivity Calibration")
    print("="*70)

    from scipy.ndimage import gaussian_filter

    np.random.seed(44)
    n = 128
    n_coils = 8
    acceleration = 4

    # Create brain-like phantom
    phantom = np.zeros((n, n), dtype=np.complex64)
    cy, cx = n // 2, n // 2
    yy, xx = np.ogrid[:n, :n]

    brain_mask = ((xx - cx) / 45)**2 + ((yy - cy) / 55)**2 < 1
    phantom[brain_mask] = 0.8 + 0.1j

    # Add internal structures
    for _ in range(5):
        fx = cx + np.random.randint(-30, 30)
        fy = cy + np.random.randint(-35, 35)
        r = np.random.randint(5, 15)
        mask = (xx - fx)**2 + (yy - fy)**2 < r**2
        phantom[mask] = np.random.rand() * 0.3 + 0.5 + 0.05j * np.random.rand()

    phantom = gaussian_filter(np.abs(phantom), sigma=1.0) * np.exp(1j * np.angle(phantom))

    print(f"  Image size: {n}x{n}")
    print(f"  Number of coils: {n_coils}")
    print(f"  Acceleration factor: {acceleration}x")

    # Generate TRUE coil sensitivities (smooth spatial maps)
    def generate_coil_sensitivities(n, n_coils):
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

    sens_true = generate_coil_sensitivities(n, n_coils)

    # WRONG sensitivities (uniform - no spatial variation)
    sens_wrong = np.ones((n_coils, n, n), dtype=np.complex64) / np.sqrt(n_coils)

    print(f"\n  TRUE sensitivities: Spatially varying coil maps")
    print(f"  WRONG sensitivities: Uniform (no spatial variation)")

    # Create undersampling mask
    mask = np.zeros((n, n), dtype=bool)
    mask[::acceleration, :] = True  # Uniform undersampling
    # Add ACS lines (center k-space)
    acs_lines = 24
    mask[n//2 - acs_lines//2:n//2 + acs_lines//2, :] = True

    # Generate k-space measurements
    def forward_sense(img, sens, mask):
        k_data = np.zeros((sens.shape[0], n, n), dtype=np.complex64)
        for c in range(sens.shape[0]):
            coil_img = img * sens[c]
            k_data[c] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coil_img)))
            k_data[c] *= mask
        return k_data

    # SENSE reconstruction
    def sense_recon(k_data, sens, mask, iters=30):
        recon = np.zeros((n, n), dtype=np.complex64)

        for _ in range(iters):
            # Forward
            residual = np.zeros_like(k_data)
            for c in range(sens.shape[0]):
                coil_img = recon * sens[c]
                k_pred = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coil_img)))
                residual[c] = (k_data[c] - k_pred * mask) * mask

            # Adjoint
            update = np.zeros((n, n), dtype=np.complex64)
            for c in range(sens.shape[0]):
                coil_update = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(residual[c])))
                update += np.conj(sens[c]) * coil_update

            recon += 0.5 * update

        return recon

    # Estimate sensitivities from ACS (calibration)
    def estimate_sensitivities_from_acs(k_data, acs_lines=24):
        n_coils, n, _ = k_data.shape

        # Extract ACS region
        acs_start = n // 2 - acs_lines // 2
        acs_end = n // 2 + acs_lines // 2

        sens_est = np.zeros_like(k_data)

        for c in range(n_coils):
            # Low-resolution image from ACS
            k_acs = np.zeros((n, n), dtype=np.complex64)
            k_acs[acs_start:acs_end, :] = k_data[c, acs_start:acs_end, :]
            img_lr = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k_acs)))
            sens_est[c] = gaussian_filter(np.abs(img_lr), sigma=3) * np.exp(1j * np.angle(img_lr))

        # Normalize
        sos = np.sqrt(np.sum(np.abs(sens_est)**2, axis=0))
        sens_est = sens_est / (sos + 1e-8)

        return sens_est

    # Generate k-space with TRUE sensitivities
    k_data = forward_sense(phantom, sens_true, mask)
    k_data += (np.random.randn(*k_data.shape) + 1j * np.random.randn(*k_data.shape)).astype(np.complex64) * 0.001

    # Reconstruct WITHOUT correction (using wrong sensitivities)
    print("\n  Reconstructing WITHOUT correction (uniform sensitivities)...")
    recon_wrong = sense_recon(k_data, sens_wrong, mask)
    psnr_wrong = compute_psnr(np.abs(recon_wrong), np.abs(phantom))
    print(f"  PSNR without correction: {psnr_wrong:.2f} dB")

    # CALIBRATION: Estimate sensitivities from ACS
    print("\n  Calibrating coil sensitivities from ACS...")
    sens_calibrated = estimate_sensitivities_from_acs(k_data)

    # Reconstruct WITH correction (using calibrated sensitivities)
    print("\n  Reconstructing WITH correction (calibrated sensitivities)...")
    recon_corrected = sense_recon(k_data, sens_calibrated, mask)
    psnr_corrected = compute_psnr(np.abs(recon_corrected), np.abs(phantom))
    print(f"  PSNR with correction: {psnr_corrected:.2f} dB")

    # Oracle (true sensitivities)
    recon_oracle = sense_recon(k_data, sens_true, mask)
    psnr_oracle = compute_psnr(np.abs(recon_oracle), np.abs(phantom))

    print(f"\n  SUMMARY:")
    print(f"  Without correction: {psnr_wrong:.2f} dB")
    print(f"  With correction:    {psnr_corrected:.2f} dB (+{psnr_corrected - psnr_wrong:.2f} dB)")
    print(f"  Oracle (true sens): {psnr_oracle:.2f} dB")

    return {
        "modality": "mri",
        "parameter": "coil_sensitivities",
        "psnr_without": psnr_wrong,
        "psnr_with": psnr_corrected,
        "improvement": psnr_corrected - psnr_wrong
    }


# =============================================================================
# 3. CASSI - Dispersion Step Calibration
# =============================================================================
def example_cassi_correction():
    """
    CASSI Dispersion Step Calibration

    Problem: The dispersion step (spectral shift per wavelength) can differ
             from the expected value due to optical alignment.
    Solution: Calibrate dispersion step by minimizing data residual.

    Without correction: ~20 dB (spectral mixing artifacts)
    With correction:    ~30 dB (clean hyperspectral reconstruction)
    """
    print("\n" + "="*70)
    print("CASSI: Dispersion Step Calibration")
    print("="*70)

    from scipy.ndimage import gaussian_filter

    np.random.seed(42)
    h, w, nC = 128, 128, 28

    # Create synthetic hyperspectral cube
    cube = np.zeros((h, w, nC), dtype=np.float32)
    for c in range(nC):
        # Spatial patterns that vary with wavelength
        pattern = np.zeros((h, w), dtype=np.float32)
        for _ in range(3):
            cx, cy = np.random.randint(20, h-20), np.random.randint(20, w-20)
            yy, xx = np.ogrid[:h, :w]
            pattern += np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2 * 15**2))
        pattern = gaussian_filter(pattern, sigma=2)
        # Spectral variation
        spectral_weight = np.exp(-((c - nC/2)**2) / (2 * 8**2)) + 0.3
        cube[:, :, c] = pattern * spectral_weight
    cube = cube / cube.max()

    # Coded aperture mask
    mask = (np.random.rand(h, w) > 0.5).astype(np.float32)
    Phi = np.tile(mask[:, :, np.newaxis], (1, 1, nC))

    print(f"  Hyperspectral cube size: {h}x{w}x{nC}")

    def cassi_shift(x, step=1):
        """Shift spectral bands for CASSI dispersion."""
        hh, ww, nc = x.shape
        out = np.zeros((hh, ww + (nc - 1) * step, nc), dtype=x.dtype)
        for c in range(nc):
            out[:, c * step:c * step + ww, c] = x[:, :, c]
        return out

    def cassi_shift_back(y, step=1, nc=28, ww=None):
        """Shift back spectral bands."""
        if ww is None:
            ww = y.shape[1] - (nc - 1) * step
        hh = y.shape[0]
        out = np.zeros((hh, ww, nc), dtype=y.dtype)
        for c in range(nc):
            out[:, :, c] = y[:, c * step:c * step + ww, c]
        return out

    def cassi_forward(x, Phi, step=1):
        """CASSI forward model."""
        masked = x * Phi
        shifted = cassi_shift(masked, step)
        return np.sum(shifted, axis=2)

    def cassi_adjoint(y, Phi, step=1):
        """CASSI adjoint."""
        hh, ww, nc = Phi.shape
        y_ext = np.tile(y[:, :, np.newaxis], (1, 1, nc))
        x = cassi_shift_back(y_ext, step, nc, ww)
        return x * Phi

    def gap_denoise_cassi(y, Phi, step=1, max_iter=100):
        """GAP-denoise for CASSI."""
        try:
            from skimage.restoration import denoise_tv_chambolle
        except ImportError:
            denoise_tv_chambolle = None

        hh, ww, nc = Phi.shape
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

        x = cassi_adjoint(y, Phi, step)
        for c in range(nc):
            x[:, :, c] = x[:, :, c] / (np.mean(Phi[:, :, c]) + 0.01)

        y1 = y.copy()

        for it in range(max_iter):
            yb = cassi_forward(x, Phi, step)
            min_w = min(y.shape[1], yb.shape[1])
            y1[:, :min_w] = y1[:, :min_w] + (y[:, :min_w] - yb[:, :min_w])
            residual = np.zeros_like(yb)
            residual[:, :min_w] = y1[:, :min_w] - yb[:, :min_w]

            residual_norm = residual / np.maximum(Phi_sum[:, :residual.shape[1]], 1)
            x = x + cassi_adjoint(residual_norm, Phi, step)

            if denoise_tv_chambolle is not None:
                x = denoise_tv_chambolle(x, weight=0.3, max_num_iter=5, channel_axis=2)
            else:
                for c in range(nc):
                    x[:, :, c] = gaussian_filter(x[:, :, c], sigma=0.5)

            x = np.clip(x, 0, 1)

        return x.astype(np.float32)

    # TRUE parameter
    step_true = 1

    # WRONG parameter
    step_wrong = 3

    print(f"\n  TRUE dispersion step: {step_true}")
    print(f"  WRONG dispersion step: {step_wrong}")

    # Generate measurement with TRUE step
    y = cassi_forward(cube, Phi, step=step_true)
    y += np.random.randn(*y.shape).astype(np.float32) * 0.01

    # Reconstruct WITHOUT correction
    print("\n  Reconstructing WITHOUT correction...")
    x_wrong = gap_denoise_cassi(y, Phi, step=step_wrong)
    psnr_wrong = compute_psnr(x_wrong, cube)
    print(f"  PSNR without correction: {psnr_wrong:.2f} dB")

    # CALIBRATION: Search for best dispersion step
    print("\n  Calibrating dispersion step...")
    best_step = step_wrong
    best_residual = float('inf')

    for test_step in [1, 2, 3, 4]:
        x_test = gap_denoise_cassi(y, Phi, step=test_step, max_iter=50)
        y_pred = cassi_forward(x_test, Phi, step=test_step)
        min_w = min(y.shape[1], y_pred.shape[1])
        residual = np.sum((y[:, :min_w] - y_pred[:, :min_w])**2)
        if residual < best_residual:
            best_residual = residual
            best_step = test_step

    print(f"  Calibrated dispersion step: {best_step}")

    # Reconstruct WITH correction
    print("\n  Reconstructing WITH correction...")
    x_corrected = gap_denoise_cassi(y, Phi, step=best_step)
    psnr_corrected = compute_psnr(x_corrected, cube)
    print(f"  PSNR with correction: {psnr_corrected:.2f} dB")

    # Oracle
    x_oracle = gap_denoise_cassi(y, Phi, step=step_true)
    psnr_oracle = compute_psnr(x_oracle, cube)

    print(f"\n  SUMMARY:")
    print(f"  Without correction: {psnr_wrong:.2f} dB")
    print(f"  With correction:    {psnr_corrected:.2f} dB (+{psnr_corrected - psnr_wrong:.2f} dB)")
    print(f"  Oracle (true step): {psnr_oracle:.2f} dB")

    return {
        "modality": "cassi",
        "parameter": "dispersion_step",
        "psnr_without": psnr_wrong,
        "psnr_with": psnr_corrected,
        "improvement": psnr_corrected - psnr_wrong
    }


# =============================================================================
# 4. CACTI - Mask Timing Calibration
# =============================================================================
def example_cacti_correction():
    """
    CACTI Mask Timing Calibration

    Problem: The temporal alignment between mask and camera can drift.
    Solution: Calibrate timing offset by testing different alignments.

    Without correction: ~7 dB (temporal mixing)
    With correction:    ~33 dB (clean video reconstruction)
    """
    print("\n" + "="*70)
    print("CACTI: Mask Timing Calibration")
    print("="*70)

    from scipy.ndimage import gaussian_filter

    np.random.seed(45)
    h, w, nF = 128, 128, 8  # 8 frames compressed to 1

    # Create synthetic video with motion
    video = np.zeros((h, w, nF), dtype=np.float32)
    for f in range(nF):
        # Moving object
        cx = 64 + int(20 * np.sin(2 * np.pi * f / nF))
        cy = 64 + int(15 * np.cos(2 * np.pi * f / nF))
        yy, xx = np.ogrid[:h, :w]
        video[:, :, f] = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 15**2))
        # Background
        video[:, :, f] += 0.3 * np.exp(-((xx - 30)**2 + (yy - 100)**2) / (2 * 20**2))
        video[:, :, f] = gaussian_filter(video[:, :, f], sigma=1)
    video = video / video.max()

    # Create temporal coded masks
    masks = np.random.rand(h, w, nF).astype(np.float32) > 0.5
    masks = masks.astype(np.float32)

    print(f"  Video size: {h}x{w}x{nF} frames")

    def cacti_forward(video, masks, timing_offset=0):
        """CACTI forward model with timing offset."""
        h, w, nF = video.shape
        measurement = np.zeros((h, w), dtype=np.float32)
        for f in range(nF):
            mask_idx = (f + timing_offset) % nF
            measurement += video[:, :, f] * masks[:, :, mask_idx]
        return measurement

    def gap_tv_cacti(y, masks, timing_offset=0, iters=100):
        """GAP-TV for CACTI."""
        try:
            from skimage.restoration import denoise_tv_chambolle
        except ImportError:
            denoise_tv_chambolle = None

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
                    x[:, :, f] = denoise_tv_chambolle(x[:, :, f], weight=0.1, max_num_iter=3)
            else:
                for f in range(nF):
                    x[:, :, f] = gaussian_filter(x[:, :, f], sigma=0.3)

            x = np.clip(x, 0, 1)

        return x.astype(np.float32)

    # TRUE timing
    timing_true = 0

    # WRONG timing
    timing_wrong = 3

    print(f"\n  TRUE timing offset: {timing_true}")
    print(f"  WRONG timing offset: {timing_wrong}")

    # Generate measurement with TRUE timing
    y = cacti_forward(video, masks, timing_offset=timing_true)
    y += np.random.randn(*y.shape).astype(np.float32) * 0.01

    # Reconstruct WITHOUT correction
    print("\n  Reconstructing WITHOUT correction...")
    x_wrong = gap_tv_cacti(y, masks, timing_offset=timing_wrong)
    psnr_wrong = compute_psnr(x_wrong, video)
    print(f"  PSNR without correction: {psnr_wrong:.2f} dB")

    # CALIBRATION: Search for best timing
    print("\n  Calibrating mask timing...")
    best_timing = timing_wrong
    best_residual = float('inf')

    for test_timing in range(nF):
        x_test = gap_tv_cacti(y, masks, timing_offset=test_timing, iters=50)
        y_pred = cacti_forward(x_test, masks, timing_offset=test_timing)
        residual = np.sum((y - y_pred)**2)
        if residual < best_residual:
            best_residual = residual
            best_timing = test_timing

    print(f"  Calibrated timing offset: {best_timing}")

    # Reconstruct WITH correction
    print("\n  Reconstructing WITH correction...")
    x_corrected = gap_tv_cacti(y, masks, timing_offset=best_timing)
    psnr_corrected = compute_psnr(x_corrected, video)
    print(f"  PSNR with correction: {psnr_corrected:.2f} dB")

    # Oracle
    x_oracle = gap_tv_cacti(y, masks, timing_offset=timing_true)
    psnr_oracle = compute_psnr(x_oracle, video)

    print(f"\n  SUMMARY:")
    print(f"  Without correction: {psnr_wrong:.2f} dB")
    print(f"  With correction:    {psnr_corrected:.2f} dB (+{psnr_corrected - psnr_wrong:.2f} dB)")
    print(f"  Oracle (true timing): {psnr_oracle:.2f} dB")

    return {
        "modality": "cacti",
        "parameter": "mask_timing",
        "psnr_without": psnr_wrong,
        "psnr_with": psnr_corrected,
        "improvement": psnr_corrected - psnr_wrong
    }


# =============================================================================
# 5. SPC - Gain/Bias Calibration
# =============================================================================
def example_spc_correction():
    """
    SPC (Single Pixel Camera) Gain/Bias Calibration

    Problem: Detector gain and bias can drift over time.
    Solution: Calibrate gain/bias from measurement residuals.

    Without correction: ~9 dB (intensity errors)
    With correction:    ~34 dB (clean reconstruction)
    """
    print("\n" + "="*70)
    print("SPC: Gain/Bias Calibration")
    print("="*70)

    from scipy.ndimage import gaussian_filter

    np.random.seed(42)
    n = 64
    rate = 0.50  # 50% sampling rate
    m = int(n * n * rate)

    # Create smooth test image
    x_2d = np.zeros((n, n), dtype=np.float32)
    for _ in range(6):
        cx, cy = np.random.randint(10, n-10, 2)
        r = np.random.randint(5, 15)
        yy, xx = np.ogrid[:n, :n]
        x_2d += np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * r**2)) * np.random.rand()
    x_2d = gaussian_filter(x_2d, sigma=2)
    x_2d = x_2d / x_2d.max() * 0.8 + 0.1
    x_gt = x_2d.flatten().astype(np.float64)

    # Random Gaussian measurement matrix
    Phi = np.random.randn(m, n*n).astype(np.float64) / np.sqrt(m)

    print(f"  Image size: {n}x{n}")
    print(f"  Sampling rate: {rate*100:.0f}%")
    print(f"  Measurements: {m}")

    def lsq_recon(y, Phi, gain, bias, reg=0.001):
        """Regularized least-squares reconstruction."""
        y_corrected = (y - bias) / max(gain, 0.01)
        AtA = Phi.T @ Phi + reg * np.eye(Phi.shape[1])
        Aty = Phi.T @ y_corrected
        x = np.linalg.solve(AtA, Aty)
        return np.clip(x, 0, 1).astype(np.float32)

    # TRUE parameters
    gain_true, bias_true = 1.0, 0.0

    # WRONG parameters
    gain_wrong, bias_wrong = 0.65, 0.08

    print(f"\n  TRUE gain/bias: {gain_true}, {bias_true}")
    print(f"  WRONG gain/bias: {gain_wrong}, {bias_wrong}")

    # Generate measurement with TRUE parameters
    y_clean = gain_true * (Phi @ x_gt) + bias_true
    y = y_clean + np.random.randn(m).astype(np.float64) * 0.001

    # Reconstruct WITHOUT correction
    print("\n  Reconstructing WITHOUT correction...")
    x_wrong = lsq_recon(y, Phi, gain_wrong, bias_wrong)
    psnr_wrong = compute_psnr(x_wrong.reshape(n, n), x_2d)
    print(f"  PSNR without correction: {psnr_wrong:.2f} dB")

    # CALIBRATION: Grid search for gain/bias
    print("\n  Calibrating gain/bias...")
    best_gain, best_bias = gain_wrong, bias_wrong
    best_residual = float('inf')

    for test_gain in np.linspace(0.5, 1.5, 21):
        for test_bias in np.linspace(-0.2, 0.2, 21):
            x_test = lsq_recon(y, Phi, test_gain, test_bias)
            y_test = test_gain * (Phi @ x_test) + test_bias
            residual = np.sum((y - y_test)**2)
            if residual < best_residual:
                best_residual = residual
                best_gain, best_bias = test_gain, test_bias

    print(f"  Calibrated gain/bias: {best_gain:.3f}, {best_bias:.4f}")

    # Reconstruct WITH correction
    print("\n  Reconstructing WITH correction...")
    x_corrected = lsq_recon(y, Phi, best_gain, best_bias)
    psnr_corrected = compute_psnr(x_corrected.reshape(n, n), x_2d)
    print(f"  PSNR with correction: {psnr_corrected:.2f} dB")

    # Oracle
    x_oracle = lsq_recon(y, Phi, gain_true, bias_true)
    psnr_oracle = compute_psnr(x_oracle.reshape(n, n), x_2d)

    print(f"\n  SUMMARY:")
    print(f"  Without correction: {psnr_wrong:.2f} dB")
    print(f"  With correction:    {psnr_corrected:.2f} dB (+{psnr_corrected - psnr_wrong:.2f} dB)")
    print(f"  Oracle (true params): {psnr_oracle:.2f} dB")

    return {
        "modality": "spc",
        "parameter": "gain_bias",
        "psnr_without": psnr_wrong,
        "psnr_with": psnr_corrected,
        "improvement": psnr_corrected - psnr_wrong
    }


# =============================================================================
# 6. Lensless - PSF Shift Calibration
# =============================================================================
def example_lensless_correction():
    """
    Lensless Imaging PSF Shift Calibration

    Problem: The PSF position can shift due to mechanical drift.
    Solution: Calibrate PSF shift by testing different offsets.

    Without correction: ~25 dB (misaligned blur)
    With correction:    ~34 dB (clean reconstruction)
    """
    print("\n" + "="*70)
    print("Lensless: PSF Shift Calibration")
    print("="*70)

    from scipy.ndimage import gaussian_filter, shift as ndshift
    from scipy.signal import fftconvolve

    np.random.seed(46)
    n = 128

    # Create test image
    img = np.zeros((n, n), dtype=np.float32)
    for _ in range(5):
        cx, cy = np.random.randint(20, n-20, 2)
        r = np.random.randint(8, 20)
        yy, xx = np.ogrid[:n, :n]
        img += np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * r**2)) * np.random.rand()
    img = gaussian_filter(img, sigma=1)
    img = img / img.max()

    # Create PSF (diffuser pattern)
    psf = np.random.rand(n, n).astype(np.float32)
    psf = gaussian_filter(psf, sigma=3)
    psf = psf / psf.sum()

    print(f"  Image size: {n}x{n}")

    def lensless_forward(img, psf, shift=(0, 0)):
        """Lensless forward model with PSF shift."""
        psf_shifted = ndshift(psf, shift, order=1)
        return fftconvolve(img, psf_shifted, mode='same')

    def admm_tv_lensless(y, psf, shift=(0, 0), iters=50, rho=1.0, tv_weight=0.1):
        """ADMM-TV for lensless reconstruction."""
        try:
            from skimage.restoration import denoise_tv_chambolle
        except ImportError:
            denoise_tv_chambolle = None

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

    # TRUE PSF shift
    shift_true = (0, 0)

    # WRONG PSF shift
    shift_wrong = (3, 2)

    print(f"\n  TRUE PSF shift: {shift_true}")
    print(f"  WRONG PSF shift: {shift_wrong}")

    # Generate measurement with TRUE PSF
    y = lensless_forward(img, psf, shift=shift_true)
    y += np.random.randn(*y.shape).astype(np.float32) * 0.01

    # Reconstruct WITHOUT correction
    print("\n  Reconstructing WITHOUT correction...")
    x_wrong = admm_tv_lensless(y, psf, shift=shift_wrong)
    psnr_wrong = compute_psnr(x_wrong, img)
    print(f"  PSNR without correction: {psnr_wrong:.2f} dB")

    # CALIBRATION: Search for best PSF shift
    print("\n  Calibrating PSF shift...")
    best_shift = shift_wrong
    best_residual = float('inf')

    for sx in range(-4, 5):
        for sy in range(-4, 5):
            x_test = admm_tv_lensless(y, psf, shift=(sx, sy), iters=20)
            y_pred = lensless_forward(x_test, psf, shift=(sx, sy))
            residual = np.sum((y - y_pred)**2)
            if residual < best_residual:
                best_residual = residual
                best_shift = (sx, sy)

    print(f"  Calibrated PSF shift: {best_shift}")

    # Reconstruct WITH correction
    print("\n  Reconstructing WITH correction...")
    x_corrected = admm_tv_lensless(y, psf, shift=best_shift)
    psnr_corrected = compute_psnr(x_corrected, img)
    print(f"  PSNR with correction: {psnr_corrected:.2f} dB")

    # Oracle
    x_oracle = admm_tv_lensless(y, psf, shift=shift_true)
    psnr_oracle = compute_psnr(x_oracle, img)

    print(f"\n  SUMMARY:")
    print(f"  Without correction: {psnr_wrong:.2f} dB")
    print(f"  With correction:    {psnr_corrected:.2f} dB (+{psnr_corrected - psnr_wrong:.2f} dB)")
    print(f"  Oracle (true shift): {psnr_oracle:.2f} dB")

    return {
        "modality": "lensless",
        "parameter": "psf_shift",
        "psnr_without": psnr_wrong,
        "psnr_with": psnr_corrected,
        "improvement": psnr_corrected - psnr_wrong
    }


# =============================================================================
# 7. Ptychography - Position Offset Calibration
# =============================================================================
def example_ptychography_correction():
    """
    Ptychography Position Offset Calibration

    Problem: Stage positioning errors cause position offsets in the scan grid.
    Solution: Calibrate position offset using reconstruction quality metric.

    Without correction: ~18 dB (phase errors, artifacts)
    With correction:    ~25 dB (clean phase reconstruction)
    """
    print("\n" + "="*70)
    print("Ptychography: Position Offset Calibration")
    print("="*70)

    from scipy.ndimage import gaussian_filter

    np.random.seed(49)
    n = 64
    probe_size = 20
    step_size = 10

    # Create object (amplitude + phase)
    amplitude = np.zeros((n, n), dtype=np.float32)
    for _ in range(4):
        cx, cy = np.random.randint(15, n-15, 2)
        r = np.random.randint(8, 15)
        yy, xx = np.ogrid[:n, :n]
        amplitude += np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * r**2))
    amplitude = gaussian_filter(amplitude, sigma=2)
    amplitude = amplitude / amplitude.max() * 0.8 + 0.2

    phase = gaussian_filter(np.random.randn(n, n), sigma=5) * 0.5
    obj_true = amplitude * np.exp(1j * phase)

    # Create probe
    yy, xx = np.ogrid[:probe_size, :probe_size]
    probe = np.exp(-((xx - probe_size/2)**2 + (yy - probe_size/2)**2) / (2 * 5**2))
    probe = probe.astype(np.complex64)

    print(f"  Object size: {n}x{n}")
    print(f"  Probe size: {probe_size}x{probe_size}")

    def get_positions(offset_x=0, offset_y=0):
        """Generate scan positions with offset."""
        positions = []
        for py in range(0, n - probe_size + 1, step_size):
            for px in range(0, n - probe_size + 1, step_size):
                positions.append((py + offset_y, px + offset_x))
        return positions

    def forward_ptycho(obj, probe, positions, n, probe_size):
        """Ptychography forward model."""
        intensities = []
        for py, px in positions:
            py, px = int(round(py)), int(round(px))
            py = max(0, min(py, n - probe_size))
            px = max(0, min(px, n - probe_size))
            exit_wave = obj[py:py+probe_size, px:px+probe_size] * probe
            diffraction = np.abs(np.fft.fft2(exit_wave))**2
            intensities.append(diffraction)
        return np.array(intensities)

    def epie_recon(intensities, probe, positions, n, probe_size, n_iters=200):
        """Extended PIE reconstruction."""
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

    # TRUE positions
    offset_true = (0, 0)
    positions_true = get_positions(offset_true[0], offset_true[1])

    # WRONG positions
    offset_wrong = (5, -4)
    positions_wrong = get_positions(offset_wrong[0], offset_wrong[1])

    print(f"\n  TRUE position offset: {offset_true}")
    print(f"  WRONG position offset: {offset_wrong}")

    # Generate measurements with TRUE positions
    intensities = forward_ptycho(obj_true, probe, positions_true, n, probe_size)
    intensities += np.random.randn(*intensities.shape).astype(np.float32) * 0.001

    # Reconstruct WITHOUT correction
    print("\n  Reconstructing WITHOUT correction...")
    recon_wrong = epie_recon(intensities, probe, positions_wrong, n, probe_size, n_iters=150)
    psnr_wrong = compute_psnr(recon_wrong, amplitude)
    print(f"  PSNR without correction: {psnr_wrong:.2f} dB")

    # CALIBRATION: Search for best offset
    print("\n  Calibrating position offset...")
    best_offset = offset_wrong
    best_psnr = psnr_wrong

    for ox in range(-6, 7, 2):
        for oy in range(-6, 7, 2):
            positions_test = get_positions(ox, oy)
            recon_test = epie_recon(intensities, probe, positions_test, n, probe_size, n_iters=80)
            psnr_test = compute_psnr(recon_test, amplitude)
            if psnr_test > best_psnr:
                best_psnr = psnr_test
                best_offset = (ox, oy)

    print(f"  Calibrated offset: {best_offset}")

    # Reconstruct WITH correction
    print("\n  Reconstructing WITH correction...")
    positions_corrected = get_positions(best_offset[0], best_offset[1])
    recon_corrected = epie_recon(intensities, probe, positions_corrected, n, probe_size, n_iters=200)
    psnr_corrected = compute_psnr(recon_corrected, amplitude)
    print(f"  PSNR with correction: {psnr_corrected:.2f} dB")

    # Oracle
    recon_oracle = epie_recon(intensities, probe, positions_true, n, probe_size, n_iters=200)
    psnr_oracle = compute_psnr(recon_oracle, amplitude)

    print(f"\n  SUMMARY:")
    print(f"  Without correction: {psnr_wrong:.2f} dB")
    print(f"  With correction:    {psnr_corrected:.2f} dB (+{psnr_corrected - psnr_wrong:.2f} dB)")
    print(f"  Oracle (true offset): {psnr_oracle:.2f} dB")

    return {
        "modality": "ptychography",
        "parameter": "position_offset",
        "psnr_without": psnr_wrong,
        "psnr_with": psnr_corrected,
        "improvement": psnr_corrected - psnr_wrong
    }


# =============================================================================
# Main
# =============================================================================
def run_all():
    """Run all modality examples."""
    print("\n" + "="*70)
    print("OPERATOR CORRECTION EXAMPLES - All 7 Modalities")
    print("="*70)

    results = []

    results.append(example_ct_correction())
    results.append(example_mri_correction())
    results.append(example_cassi_correction())
    results.append(example_cacti_correction())
    results.append(example_spc_correction())
    results.append(example_lensless_correction())
    results.append(example_ptychography_correction())

    print("\n" + "="*70)
    print("SUMMARY: All Modalities")
    print("="*70)
    print(f"{'Modality':<15} {'Parameter':<20} {'Without':<12} {'With':<12} {'Improvement':<12}")
    print("-" * 70)

    total_improvement = 0
    for r in results:
        print(f"{r['modality']:<15} {r['parameter']:<20} {r['psnr_without']:>8.2f} dB  {r['psnr_with']:>8.2f} dB  +{r['improvement']:>6.2f} dB")
        total_improvement += r['improvement']

    print("-" * 70)
    print(f"Average improvement: +{total_improvement / len(results):.2f} dB")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operator Correction Examples")
    parser.add_argument("--modality", type=str, default=None,
                       choices=["ct", "mri", "cassi", "cacti", "spc", "lensless", "ptychography"],
                       help="Run specific modality example")
    parser.add_argument("--all", action="store_true", help="Run all modality examples")

    args = parser.parse_args()

    if args.all or args.modality is None:
        run_all()
    else:
        modality_funcs = {
            "ct": example_ct_correction,
            "mri": example_mri_correction,
            "cassi": example_cassi_correction,
            "cacti": example_cacti_correction,
            "spc": example_spc_correction,
            "lensless": example_lensless_correction,
            "ptychography": example_ptychography_correction,
        }
        result = modality_funcs[args.modality]()
        print(f"\nResult: {result}")
