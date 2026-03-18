"""Light-Sheet Microscopy Destriping: Fourier Notch + VSNR.

Directional stripe artifact removal for light-sheet fluorescence microscopy.

References:
- Fehrenbach, J. et al. (2012). "Variational algorithms to remove stationary
  noise: applications to microscopy imaging", IEEE TIP. (VSNR)

Benchmark:
- VSNR: effective stripe removal with minimal structure loss
- CPU-only, no GPU needed
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Algorithm 1: Fourier Notch Destriping
# ---------------------------------------------------------------------------

def fourier_notch_destripe(
    image: np.ndarray,
    stripe_direction: str = "horizontal",
    notch_width: int = 3,
    damping: float = 0.0,
) -> np.ndarray:
    """FFT-based notch filter for stripe removal.

    Removes stripe artifacts by masking out frequencies concentrated along
    the stripe direction in Fourier space.

    For horizontal stripes the energy sits along the ky axis (kx ~ 0);
    masking a narrow band around kx = 0 (excluding DC) suppresses stripes
    while preserving most image structure.

    Args:
        image: Input image (H, W), float32.
        stripe_direction: 'horizontal' or 'vertical'.
        notch_width: Half-width of the notch band in pixels.
        damping: If > 0, use a soft Gaussian roll-off instead of a hard mask.
                 Value is the sigma of the Gaussian taper.

    Returns:
        Destriped image (H, W), float32.
    """
    image = image.astype(np.float32)
    h, w = image.shape

    F = fftshift(fft2(image))

    cy, cx = h // 2, w // 2

    # Build the notch mask (1 = keep, 0 = suppress)
    mask = np.ones((h, w), dtype=np.float32)

    if stripe_direction == "horizontal":
        # Horizontal stripes -> energy on ky axis (kx ~ 0)
        # Mask a vertical strip in Fourier space centred on kx = 0
        if damping > 0:
            x_coords = np.arange(w) - cx
            taper = 1.0 - np.exp(-x_coords ** 2 / (2 * damping ** 2))
            mask *= taper[np.newaxis, :]
        else:
            lo = max(cx - notch_width, 0)
            hi = min(cx + notch_width + 1, w)
            mask[:, lo:hi] = 0.0
    else:
        # Vertical stripes -> energy on kx axis (ky ~ 0)
        if damping > 0:
            y_coords = np.arange(h) - cy
            taper = 1.0 - np.exp(-y_coords ** 2 / (2 * damping ** 2))
            mask *= taper[:, np.newaxis]
        else:
            lo = max(cy - notch_width, 0)
            hi = min(cy + notch_width + 1, h)
            mask[lo:hi, :] = 0.0

    # Preserve DC component
    mask[cy, cx] = 1.0

    # Apply mask and reconstruct
    F_filtered = F * mask
    result = np.real(ifft2(ifftshift(F_filtered)))

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Algorithm 2: Wavelet Destriping
# ---------------------------------------------------------------------------

def wavelet_destripe(
    image: np.ndarray,
    level: int = 4,
    sigma: float = 2.0,
) -> np.ndarray:
    """Wavelet-based destriping.

    Decomposes the image using a multi-level wavelet transform, applies
    Gaussian smoothing to the horizontal detail coefficients to suppress
    stripe energy, then reconstructs.

    Falls back to a simpler spatial-domain approach if PyWavelets
    (pywt) is not available.

    Args:
        image: Input image (H, W), float32.
        level: Number of wavelet decomposition levels.
        sigma: Gaussian sigma applied along the stripe direction in each
               detail sub-band.

    Returns:
        Destriped image (H, W), float32.
    """
    image = image.astype(np.float32)

    try:
        import pywt

        coeffs = pywt.wavedec2(image, "db4", level=level)

        # coeffs = [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
        # Horizontal detail coefficients (cH) carry most stripe energy.
        new_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            cH, cV, cD = detail
            # Smooth cH along columns (axis 1) to kill horizontal stripe
            cH_smooth = gaussian_filter1d(cH, sigma=sigma, axis=1)
            new_coeffs.append((cH_smooth, cV, cD))

        result = pywt.waverec2(new_coeffs, "db4")

        # waverec2 may produce slightly different size; crop to match.
        result = result[: image.shape[0], : image.shape[1]]

    except ImportError:
        # Fallback: directional Gaussian filtering in spatial domain.
        # Estimate stripe component with column-wise mean, then subtract.
        stripe_profile = np.mean(image, axis=1, keepdims=True)
        stripe_profile_smooth = gaussian_filter1d(
            stripe_profile, sigma=sigma * 4, axis=0
        )
        stripe_component = stripe_profile - stripe_profile_smooth
        result = image - stripe_component

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Algorithm 3: VSNR Destriping (Variational Stripe Noise Removal)
# ---------------------------------------------------------------------------

def _gradient_operator(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete gradient of a 2-D image.

    Returns:
        (Dx, Dy) -- horizontal and vertical finite differences.
    """
    dx = np.diff(x, axis=1, prepend=x[:, :1])
    dy = np.diff(x, axis=0, prepend=x[:1, :])
    return dx, dy


def _divergence_operator(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Negative adjoint of the gradient (2-D divergence)."""
    div_x = np.diff(px, axis=1, append=px[:, -1:])
    div_y = np.diff(py, axis=0, append=py[-1:, :])
    return -(div_x + div_y)


def _build_stripe_basis(
    shape: Tuple[int, int],
    directions: list,
) -> list:
    """Build directional stripe basis patterns D_k.

    Each pattern D_k is a rank-1 image whose columns are constant
    (for horizontal stripes at angle 0) or rotated equivalently.

    Args:
        shape: (H, W) of the image.
        directions: list of angles in degrees.  0 = horizontal stripes.

    Returns:
        List of 2-D arrays, one per direction.
    """
    h, w = shape
    patterns = []

    for angle_deg in directions:
        angle_rad = np.deg2rad(angle_deg)

        # Create coordinate grid
        y_coords = np.arange(h) - h / 2.0
        x_coords = np.arange(w) - w / 2.0
        Y, X = np.meshgrid(y_coords, x_coords, indexing="ij")

        # Project onto the stripe-perpendicular direction
        t = X * np.sin(angle_rad) + Y * np.cos(angle_rad)

        # The basis vector is a cosine along the perpendicular direction
        # We use a Dirac-comb style basis (constant along stripe direction)
        # Simplified: for angle 0, every row is a constant -> rank-1
        if abs(angle_deg % 180) < 1e-6:
            # Purely horizontal stripes: constant across columns
            basis = np.ones((h, w), dtype=np.float32)
            basis *= np.arange(h, dtype=np.float32).reshape(-1, 1) / h
        elif abs((angle_deg - 90) % 180) < 1e-6:
            # Purely vertical stripes
            basis = np.ones((h, w), dtype=np.float32)
            basis *= np.arange(w, dtype=np.float32).reshape(1, -1) / w
        else:
            # General angle
            basis = t / (np.max(np.abs(t)) + 1e-10)

        patterns.append(basis.astype(np.float32))

    return patterns


def vsnr_destripe(
    image: np.ndarray,
    lam: float = 1.0,
    directions: Optional[list] = None,
    iters: int = 100,
    tol: float = 1e-5,
) -> np.ndarray:
    """Variational Stripe Noise Removal (VSNR).

    Model: y = x + sum_k(D_k * n_k) where D_k are directional stripe
    basis patterns and n_k are stripe coefficient images.

    Solves via ADMM:
        min_x  TV(x)  +  (lam / 2) * || y - x - sum_k D_k n_k ||^2

    The TV term preserves edges while the data-fidelity drives stripe
    fitting through the directional basis.

    Args:
        image: Input image (H, W), float32.
        lam: Regularisation weight (higher = stronger data fidelity,
             weaker destriping).
        directions: List of stripe angles in degrees.
                    Default [0] for horizontal stripes.
        iters: Maximum ADMM iterations.
        tol: Convergence tolerance on relative change in x.

    Returns:
        Destriped image (H, W), float32.
    """
    image = image.astype(np.float64)
    h, w = image.shape

    if directions is None:
        directions = [0]

    # Build stripe basis patterns
    patterns = _build_stripe_basis((h, w), directions)
    n_patterns = len(patterns)

    # ADMM parameters
    rho = 1.0  # Augmented Lagrangian penalty

    # Initialise primal variables
    x = image.copy()
    n_coeffs = [np.zeros((h, w), dtype=np.float64) for _ in range(n_patterns)]

    # Auxiliary variable for TV splitting: z = grad(x)
    zx = np.zeros((h, w), dtype=np.float64)
    zy = np.zeros((h, w), dtype=np.float64)

    # Dual variables
    ux = np.zeros((h, w), dtype=np.float64)
    uy = np.zeros((h, w), dtype=np.float64)

    for iteration in range(iters):
        x_prev = x.copy()

        # ------------------------------------------------------------------
        # Step 1: Update stripe coefficients n_k (closed-form per pattern)
        # n_k = D_k^T (y - x) / (||D_k||^2 + eps)
        # ------------------------------------------------------------------
        residual = image - x
        for k in range(n_patterns):
            Dk = patterns[k]
            dk_norm_sq = np.sum(Dk ** 2) + 1e-10
            n_coeffs[k] = (Dk * residual) / dk_norm_sq

        # Compute stripe estimate
        stripe = np.zeros((h, w), dtype=np.float64)
        for k in range(n_patterns):
            stripe += patterns[k] * n_coeffs[k]

        # ------------------------------------------------------------------
        # Step 2: Update x via Fourier solve
        # (lam I + rho * nabla^T nabla) x = lam (y - stripe) + rho * div(z - u)
        # ------------------------------------------------------------------
        rhs = lam * (image - stripe) + rho * _divergence_operator(
            zx - ux, zy - uy
        )

        # Solve in Fourier domain using eigenvalues of Laplacian
        # eigenvalues of -nabla^T nabla
        fy = np.fft.fftfreq(h).reshape(-1, 1)
        fx = np.fft.fftfreq(w).reshape(1, -1)
        lap_eigenvalues = (
            2 * (1 - np.cos(2 * np.pi * fy))
            + 2 * (1 - np.cos(2 * np.pi * fx))
        )

        RHS_hat = fft2(rhs)
        denom = lam + rho * lap_eigenvalues
        denom[0, 0] = max(denom[0, 0], 1e-10)  # Avoid division by zero

        x = np.real(ifft2(RHS_hat / denom))

        # ------------------------------------------------------------------
        # Step 3: Update z (TV proximal / shrinkage on gradient)
        # z = shrink(grad(x) + u, 1/rho)
        # ------------------------------------------------------------------
        dx, dy = _gradient_operator(x)

        vx = dx + ux
        vy = dy + uy

        # Isotropic TV shrinkage
        mag = np.sqrt(vx ** 2 + vy ** 2)
        shrink_factor = np.maximum(mag - 1.0 / rho, 0) / (mag + 1e-10)

        zx = shrink_factor * vx
        zy = shrink_factor * vy

        # ------------------------------------------------------------------
        # Step 4: Update dual variables
        # ------------------------------------------------------------------
        ux += dx - zx
        uy += dy - zy

        # ------------------------------------------------------------------
        # Convergence check
        # ------------------------------------------------------------------
        change = np.linalg.norm(x - x_prev) / (np.linalg.norm(x) + 1e-10)
        if change < tol:
            break

    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Portfolio wrapper
# ---------------------------------------------------------------------------

def run_lightsheet(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run light-sheet destriping reconstruction.

    Selects among Fourier notch, wavelet, and VSNR destriping based on
    ``cfg["solver"]``.  Default is VSNR.

    Args:
        y: Striped light-sheet image (H, W) or stack (N, H, W).
        physics: LightSheet physics operator (used for metadata only).
        cfg: Configuration dict with:
            - solver: 'fourier_notch', 'wavelet', or 'vsnr' (default 'vsnr')
            - stripe_direction: 'horizontal' or 'vertical' (default 'horizontal')
            - notch_width: For Fourier notch (default 3)
            - damping: Soft roll-off sigma for Fourier notch (default 0.0)
            - level: Wavelet decomposition levels (default 4)
            - sigma: Gaussian sigma for wavelet method (default 2.0)
            - lam: VSNR regularisation weight (default 1.0)
            - directions: VSNR stripe angles in degrees (default [0])
            - iters: VSNR iterations (default 100)

    Returns:
        Tuple of (destriped_image, info_dict)
    """
    solver = cfg.get("solver", "vsnr")
    stripe_direction = cfg.get("stripe_direction", "horizontal")

    info: Dict[str, Any] = {
        "solver": solver,
        "stripe_direction": stripe_direction,
    }

    try:
        # Extract physics metadata if available
        if hasattr(physics, "info"):
            op_info = physics.info()
            stripe_direction = op_info.get("stripe_direction", stripe_direction)
            info["stripe_direction"] = stripe_direction

        if hasattr(physics, "stripe_direction"):
            stripe_direction = physics.stripe_direction
            info["stripe_direction"] = stripe_direction

        # Handle stack input: process each slice independently
        if y.ndim == 3:
            slices = []
            for i in range(y.shape[0]):
                s, _ = run_lightsheet(y[i], physics, cfg)
                slices.append(s)
            result = np.stack(slices, axis=0)
            info["n_slices"] = y.shape[0]
            return result, info

        # Ensure 2-D input
        if y.ndim != 2:
            info["error"] = "unexpected_input_shape"
            return y.astype(np.float32), info

        # Dispatch to chosen solver
        if solver == "fourier_notch":
            notch_width = cfg.get("notch_width", 3)
            damping = cfg.get("damping", 0.0)
            result = fourier_notch_destripe(
                y,
                stripe_direction=stripe_direction,
                notch_width=notch_width,
                damping=damping,
            )
            info["notch_width"] = notch_width
            info["damping"] = damping

        elif solver == "wavelet":
            level = cfg.get("level", 4)
            sigma = cfg.get("sigma", 2.0)
            result = wavelet_destripe(y, level=level, sigma=sigma)
            info["level"] = level
            info["sigma"] = sigma

        elif solver == "vsnr":
            lam = cfg.get("lam", 1.0)
            directions = cfg.get("directions", [0])
            iters = cfg.get("iters", 100)
            result = vsnr_destripe(
                y, lam=lam, directions=directions, iters=iters
            )
            info["lam"] = lam
            info["directions"] = directions
            info["iters"] = iters

        else:
            info["error"] = f"unknown solver: {solver}"
            return y.astype(np.float32), info

        return result, info

    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
