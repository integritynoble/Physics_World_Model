"""Panorama / Multi-Focus Image Fusion Solvers.

Traditional CPU-based algorithms for multi-focus image fusion and
panoramic stitching with multi-resolution blending.

References:
- Burt, P. & Adelson, E. (1983). "A Multiresolution Spline with
  Application to Image Mosaics", ACM ToG.
- He, K. et al. (2013). "Guided Image Filtering", IEEE TPAMI.

All methods are CPU-only (no GPU/PyTorch required).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# Laplacian pyramid utilities
# ===========================================================================


def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """Create 1D Gaussian kernel."""
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Separable Gaussian blur (CPU).

    Args:
        img: 2D or 3D image (H, W) or (H, W, C).
        sigma: Gaussian standard deviation.

    Returns:
        Blurred image, same shape.
    """
    from scipy.ndimage import gaussian_filter
    if img.ndim == 2:
        return gaussian_filter(img.astype(np.float32), sigma=sigma)
    # Channel-wise
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        out[:, :, c] = gaussian_filter(img[:, :, c].astype(np.float32), sigma=sigma)
    return out


def _downsample(img: np.ndarray) -> np.ndarray:
    """Downsample by 2x with Gaussian anti-aliasing."""
    blurred = gaussian_blur(img, sigma=1.0)
    return blurred[::2, ::2]


def _upsample(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Upsample to target spatial shape using bilinear interpolation."""
    th, tw = target_shape[:2]
    if img.ndim == 2:
        from scipy.ndimage import zoom
        factors = (th / img.shape[0], tw / img.shape[1])
        return zoom(img.astype(np.float32), factors, order=1)
    # Color image
    from scipy.ndimage import zoom
    factors = (th / img.shape[0], tw / img.shape[1], 1)
    return zoom(img.astype(np.float32), factors, order=1)


def build_gaussian_pyramid(
    img: np.ndarray, n_levels: int = 5
) -> List[np.ndarray]:
    """Build Gaussian pyramid.

    Args:
        img: Input image (H, W) or (H, W, C).
        n_levels: Number of pyramid levels.

    Returns:
        List of images from fine to coarse.
    """
    pyramid = [img.astype(np.float32)]
    for _ in range(n_levels - 1):
        pyramid.append(_downsample(pyramid[-1]))
    return pyramid


def build_laplacian_pyramid(
    img: np.ndarray, n_levels: int = 5
) -> List[np.ndarray]:
    """Build Laplacian pyramid.

    Args:
        img: Input image.
        n_levels: Number of levels.

    Returns:
        List of Laplacian images (fine to coarse), last is Gaussian residual.
    """
    gauss = build_gaussian_pyramid(img, n_levels)
    lap = []
    for i in range(len(gauss) - 1):
        upsampled = _upsample(gauss[i + 1], gauss[i].shape)
        lap.append(gauss[i] - upsampled)
    lap.append(gauss[-1])  # Coarsest level is the residual
    return lap


def reconstruct_from_laplacian(pyramid: List[np.ndarray]) -> np.ndarray:
    """Reconstruct image from Laplacian pyramid.

    Args:
        pyramid: Laplacian pyramid (fine to coarse).

    Returns:
        Reconstructed image.
    """
    img = pyramid[-1].astype(np.float32)
    for i in range(len(pyramid) - 2, -1, -1):
        img = _upsample(img, pyramid[i].shape) + pyramid[i]
    return img


# ===========================================================================
# Multi-focus fusion
# ===========================================================================


def focus_measure(img: np.ndarray, method: str = "laplacian") -> np.ndarray:
    """Compute per-pixel focus measure (sharpness map).

    Args:
        img: Grayscale image (H, W), float32.
        method: Focus measure type ('laplacian', 'gradient', 'variance').

    Returns:
        Focus map (H, W), higher = sharper.
    """
    if method == "laplacian":
        # Laplacian magnitude (edge/focus response)
        from scipy.ndimage import laplace
        return np.abs(laplace(img.astype(np.float64))).astype(np.float32)

    elif method == "gradient":
        gy, gx = np.gradient(img.astype(np.float64))
        return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

    elif method == "variance":
        # Local variance in a window
        from scipy.ndimage import uniform_filter
        mean = uniform_filter(img.astype(np.float64), size=9)
        mean_sq = uniform_filter(img.astype(np.float64) ** 2, size=9)
        return np.maximum(mean_sq - mean ** 2, 0).astype(np.float32)

    raise ValueError(f"Unknown focus measure: {method}")


def multifocus_fusion_laplacian(
    images: List[np.ndarray],
    n_levels: int = 5,
    focus_method: str = "laplacian",
    blur_sigma: float = 5.0,
) -> np.ndarray:
    """Multi-focus image fusion using Laplacian pyramid blending.

    Selects the sharpest pixels from multiple focal-plane images and
    blends them using multi-resolution pyramid decomposition.

    Args:
        images: List of co-registered images at different focal planes.
            Each (H, W) grayscale or (H, W, C) color, float32 in [0, 1].
        n_levels: Number of pyramid levels.
        focus_method: Focus measure ('laplacian', 'gradient', 'variance').
        blur_sigma: Sigma for smoothing the decision map.

    Returns:
        All-in-focus fused image, same shape as inputs.
    """
    n_images = len(images)
    if n_images == 0:
        raise ValueError("At least one image required")
    if n_images == 1:
        return images[0].copy()

    # Convert to grayscale for focus computation
    grays = []
    for img in images:
        if img.ndim == 3:
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            gray = img
        grays.append(gray.astype(np.float32))

    # Compute focus maps
    focus_maps = [focus_measure(g, focus_method) for g in grays]

    # Smooth focus maps
    smooth_maps = [gaussian_blur(fm, sigma=blur_sigma) for fm in focus_maps]

    # Create binary decision maps (argmax per pixel)
    stacked = np.stack(smooth_maps, axis=0)  # (N, H, W)
    winner = np.argmax(stacked, axis=0)  # (H, W)

    # Create weight maps
    weight_maps = []
    for i in range(n_images):
        w = (winner == i).astype(np.float32)
        # Smooth weight map transitions
        w = gaussian_blur(w, sigma=blur_sigma / 2)
        weight_maps.append(w)

    # Normalize weights (add small epsilon to avoid division by zero)
    w_sum = sum(weight_maps)
    w_sum = np.maximum(w_sum, 1e-8)
    weight_maps = [w / w_sum for w in weight_maps]

    # Build Laplacian pyramids for each image
    img_pyramids = [build_laplacian_pyramid(img, n_levels) for img in images]
    weight_pyramids = [build_gaussian_pyramid(w, n_levels) for w in weight_maps]

    # Blend at each level
    fused_pyramid = []
    for level in range(n_levels):
        blended = np.zeros_like(img_pyramids[0][level])
        for i in range(n_images):
            w = weight_pyramids[i][level]
            if blended.ndim == 3 and w.ndim == 2:
                w = w[:, :, np.newaxis]
            blended += w * img_pyramids[i][level]
        fused_pyramid.append(blended)

    # Reconstruct
    result = reconstruct_from_laplacian(fused_pyramid)
    return np.clip(result, 0, 1).astype(np.float32)


# ===========================================================================
# Guided filter fusion (alternative)
# ===========================================================================


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 0.01,
) -> np.ndarray:
    """Guided image filter (He et al. 2013).

    Args:
        guide: Guide image (H, W), float32.
        src: Source image to filter (H, W), float32.
        radius: Filter radius (box size = 2*radius + 1).
        eps: Regularization parameter.

    Returns:
        Filtered image (H, W).
    """
    from scipy.ndimage import uniform_filter
    size = 2 * radius + 1

    mean_I = uniform_filter(guide.astype(np.float64), size)
    mean_p = uniform_filter(src.astype(np.float64), size)
    mean_Ip = uniform_filter((guide * src).astype(np.float64), size)
    mean_II = uniform_filter((guide * guide).astype(np.float64), size)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, size)
    mean_b = uniform_filter(b, size)

    return (mean_a * guide + mean_b).astype(np.float32)


def multifocus_fusion_guided(
    images: List[np.ndarray],
    radius: int = 8,
    eps: float = 0.01,
) -> np.ndarray:
    """Multi-focus fusion using guided filter for weight refinement.

    Args:
        images: List of co-registered focal-plane images.
        radius: Guided filter radius.
        eps: Guided filter regularization.

    Returns:
        All-in-focus fused image.
    """
    n_images = len(images)
    if n_images == 0:
        raise ValueError("At least one image required")
    if n_images == 1:
        return images[0].copy()

    # Grayscale for focus
    grays = []
    for img in images:
        if img.ndim == 3:
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            gray = img
        grays.append(gray.astype(np.float32))

    # Focus maps
    focus_maps = [focus_measure(g, "laplacian") for g in grays]

    # Binary decision
    stacked = np.stack(focus_maps, axis=0)
    winner = np.argmax(stacked, axis=0)

    # Refine weights with guided filter
    weight_maps = []
    for i in range(n_images):
        w = (winner == i).astype(np.float32)
        w_refined = guided_filter(grays[i], w, radius=radius, eps=eps)
        weight_maps.append(np.clip(w_refined, 0, 1))

    # Normalize
    w_sum = sum(weight_maps)
    w_sum = np.maximum(w_sum, 1e-10)
    weight_maps = [w / w_sum for w in weight_maps]

    # Blend
    result = np.zeros_like(images[0], dtype=np.float32)
    for i in range(n_images):
        w = weight_maps[i]
        if result.ndim == 3 and w.ndim == 2:
            w = w[:, :, np.newaxis]
        result += w * images[i].astype(np.float32)

    return np.clip(result, 0, 1).astype(np.float32)


# ===========================================================================
# Portfolio wrapper
# ===========================================================================


def run_panorama_fusion(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for panorama / multi-focus fusion.

    Args:
        y: Either a single image or a stack of images.
            If (N, H, W) or (N, H, W, C), treated as N focal planes.
            If (H, W) or (H, W, C), returned as-is.
        physics: Physics operator (used to extract focal stack info).
        cfg: Configuration with optional keys:
            - method: 'laplacian' or 'guided' (default 'laplacian').
            - n_levels: Pyramid levels (default 5).
            - focus_method: Focus measure type (default 'laplacian').
            - blur_sigma: Weight map smoothing (default 5.0).
            - guided_radius: Guided filter radius (default 8).
            - guided_eps: Guided filter eps (default 0.01).

    Returns:
        Tuple of (fused_image, info_dict).
    """
    method = cfg.get("method", "laplacian")

    info: Dict[str, Any] = {
        "solver": "panorama_fusion",
        "method": method,
    }

    try:
        # Extract image stack
        images: List[np.ndarray] = []

        if hasattr(physics, "focal_stack"):
            images = [img.astype(np.float32) for img in physics.focal_stack]
        elif hasattr(physics, "images"):
            images = [img.astype(np.float32) for img in physics.images]
        elif y.ndim >= 3 and y.shape[0] > 1 and y.ndim <= 4:
            # Treat first dim as stack
            if y.ndim == 3:
                # Could be (N, H, W) stack or (H, W, C) color
                if y.shape[2] <= 4:
                    # Likely single color image
                    images = [y.astype(np.float32)]
                else:
                    images = [y[i].astype(np.float32) for i in range(y.shape[0])]
            else:
                images = [y[i].astype(np.float32) for i in range(y.shape[0])]
        else:
            images = [y.astype(np.float32)]

        if len(images) <= 1:
            info["note"] = "single image, no fusion needed"
            return images[0] if images else y.astype(np.float32), info

        info["n_images"] = len(images)

        if method == "guided":
            result = multifocus_fusion_guided(
                images,
                radius=cfg.get("guided_radius", 8),
                eps=cfg.get("guided_eps", 0.01),
            )
        else:
            result = multifocus_fusion_laplacian(
                images,
                n_levels=cfg.get("n_levels", 5),
                focus_method=cfg.get("focus_method", "laplacian"),
                blur_sigma=cfg.get("blur_sigma", 5.0),
            )

        return result, info

    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
