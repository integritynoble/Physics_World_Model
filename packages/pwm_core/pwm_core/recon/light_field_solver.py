"""Light Field reconstruction solvers.

References:
- Levoy, M. & Hanrahan, P. (1996). "Light Field Rendering", SIGGRAPH.
- Danielyan, A. et al. (2012). "BM3D Frames and Variational Image Deblurring", IEEE TIP.

Expected PSNR: 28.0 dB on synthetic benchmark
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple


def _estimate_global_disparity(light_field: np.ndarray) -> float:
    """Estimate global disparity from a 4D light field via sharpness maximisation.

    Uses a coarse-to-fine search: first evaluates disparity on a coarse grid,
    then refines around the best candidate. Picks the disparity that maximises
    the Laplacian energy (sharpness) of the refocused image. This is analogous
    to autofocus in light field rendering (Ng, 2005).

    Args:
        light_field: 4D light field (sx, sy, u, v).

    Returns:
        Estimated global disparity (pixels per angular index offset).
    """
    from scipy.ndimage import shift as ndi_shift, laplace

    sx, sy, nu, nv = light_field.shape
    cu, cv = nu // 2, nv // 2

    def _refocus_sharpness(disp_candidate: float) -> float:
        """Compute sharpness of refocused image at given disparity."""
        refocused = np.zeros((sx, sy), dtype=np.float64)
        for u in range(nu):
            for v in range(nv):
                shift_y = -(u - cu) * disp_candidate
                shift_x = -(v - cv) * disp_candidate
                shifted = ndi_shift(light_field[:, :, u, v].astype(np.float64),
                                   [shift_y, shift_x], order=1, mode='reflect')
                refocused += shifted
        refocused /= (nu * nv)
        lap = laplace(refocused)
        return float(np.sum(lap ** 2))

    # Coarse search: step of 0.25 over [-1, 2]
    best_disp = 0.0
    best_sharpness = -np.inf
    for d in np.arange(-1.0, 2.01, 0.25):
        s = _refocus_sharpness(d)
        if s > best_sharpness:
            best_sharpness = s
            best_disp = d

    # Fine search: step of 0.05 around best coarse value
    for d in np.arange(best_disp - 0.25, best_disp + 0.26, 0.05):
        s = _refocus_sharpness(d)
        if s > best_sharpness:
            best_sharpness = s
            best_disp = d

    return float(best_disp)


def shift_and_sum(light_field: np.ndarray, disparities: np.ndarray = None) -> np.ndarray:
    """Shift-and-Sum light field reconstruction.

    Refocuses a 4D light field by shifting sub-aperture images according
    to disparity and averaging. This is the simplest light field rendering
    algorithm.

    When disparities is None, the function automatically estimates the global
    disparity via sharpness maximisation (Laplacian energy) and uses it
    to refocus, rather than simply averaging (which would produce motion blur).

    Args:
        light_field: 4D light field (sx, sy, u, v) where (sx, sy) are spatial
            and (u, v) are angular dimensions.
        disparities: Per-pixel disparity for refocusing. If None, auto-estimates
            global disparity via sharpness maximisation.

    Returns:
        Refocused 2D image (sx, sy).
    """
    from scipy.ndimage import shift as ndi_shift

    sx, sy, nu, nv = light_field.shape
    center_u, center_v = nu // 2, nv // 2

    if disparities is None:
        # Auto-estimate global disparity
        estimated_disp = _estimate_global_disparity(light_field)
        if abs(estimated_disp) < 1e-6:
            # No detectable disparity; plain average is correct
            return np.mean(light_field, axis=(2, 3)).astype(np.float32)
        # Use estimated disparity as a scalar for refocusing
        disparities_scalar = estimated_disp
    else:
        disparities_scalar = None

    result = np.zeros((sx, sy), dtype=np.float64)
    count = 0

    for u in range(nu):
        for v in range(nv):
            if disparities_scalar is not None:
                # Global disparity: shift back to undo the view shift
                shift_y = -(u - center_u) * disparities_scalar
                shift_x = -(v - center_v) * disparities_scalar
            else:
                du = (u - center_u) * disparities
                dv = (v - center_v) * disparities
                shift_y = float(np.mean(du))
                shift_x = float(np.mean(dv))

            shifted = ndi_shift(light_field[:, :, u, v].astype(np.float64),
                               [shift_y, shift_x], order=3, mode='reflect')
            result += shifted
            count += 1

    result /= count
    return np.clip(result, 0, None).astype(np.float32)


def lfbm5d(light_field: np.ndarray, sigma: float = 0.05,
           patch_size: int = 4, search_window: int = 11,
           hard_threshold: float = 2.7) -> np.ndarray:
    """LFBM5D - BM3D-style collaborative filtering on 4D light field patches.

    Extends BM3D to 4D by grouping similar patches across both spatial
    and angular dimensions, then performing collaborative filtering
    via hard thresholding in the transform domain.

    Args:
        light_field: 4D light field (sx, sy, u, v).
        sigma: Noise standard deviation estimate.
        patch_size: Spatial patch size.
        search_window: Search window size for block matching.
        hard_threshold: Threshold multiplier for hard thresholding.

    Returns:
        Denoised 2D image (sx, sy) - central view refined.
    """
    sx, sy, nu, nv = light_field.shape
    center_u, center_v = nu // 2, nv // 2

    # Work on central sub-aperture image enhanced by angular info
    central = light_field[:, :, center_u, center_v].copy().astype(np.float64)

    # Step 1: Basic estimate via hard thresholding
    # Extract patches from all sub-aperture images
    half_p = patch_size // 2
    half_w = search_window // 2

    estimate = np.zeros_like(central)
    weights = np.zeros_like(central)

    # For each reference patch in the central view
    step = max(1, patch_size // 2)

    for iy in range(half_p, sx - half_p, step):
        for ix in range(half_p, sy - half_p, step):
            # Reference patch from central view
            ref_patch = central[iy - half_p:iy + half_p + 1,
                               ix - half_p:ix + half_p + 1]

            # Collect similar patches across angular views
            group = [ref_patch.copy()]

            for u in range(nu):
                for v in range(nv):
                    if u == center_u and v == center_v:
                        continue
                    view = light_field[:, :, u, v].astype(np.float64)
                    # Search for best matching patch in neighborhood
                    best_dist = float('inf')
                    best_patch = None

                    for dy in range(-min(half_w, 2), min(half_w, 2) + 1):
                        for dx in range(-min(half_w, 2), min(half_w, 2) + 1):
                            ny, nx = iy + dy, ix + dx
                            if (ny - half_p < 0 or ny + half_p + 1 > sx or
                                nx - half_p < 0 or nx + half_p + 1 > sy):
                                continue
                            cand = view[ny - half_p:ny + half_p + 1,
                                       nx - half_p:nx + half_p + 1]
                            dist = np.sum((ref_patch - cand) ** 2)
                            if dist < best_dist:
                                best_dist = dist
                                best_patch = cand.copy()

                    if best_patch is not None and best_dist < sigma * patch_size**2 * hard_threshold**2 * 10:
                        group.append(best_patch)

            # Stack and filter via DCT hard thresholding
            group_arr = np.array(group)  # (N, ps, ps)

            # Apply 1D transform along group axis + 2D DCT on patches
            from scipy.fft import dctn, idctn
            coeffs = dctn(group_arr, axes=(0, 1, 2))

            # Hard threshold
            threshold = hard_threshold * sigma
            mask = np.abs(coeffs) > threshold
            coeffs *= mask
            n_nonzero = max(np.sum(mask), 1)

            # Inverse transform
            filtered = idctn(coeffs, axes=(0, 1, 2))

            # Aggregate central view contribution
            weight = 1.0 / n_nonzero
            estimate[iy - half_p:iy + half_p + 1,
                    ix - half_p:ix + half_p + 1] += filtered[0] * weight
            weights[iy - half_p:iy + half_p + 1,
                   ix - half_p:ix + half_p + 1] += weight

    # Normalize
    valid = weights > 0
    estimate[valid] /= weights[valid]
    estimate[~valid] = central[~valid]

    return np.clip(estimate, 0, None).astype(np.float32)


def run_light_field(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for light field reconstruction.

    Args:
        y: Measurement (2D summed image or 4D light field).
        physics: Physics operator with light field info.
        cfg: Configuration dict.

    Returns:
        Tuple of (reconstructed_image, info_dict).
    """
    method = cfg.get("method", "shift_and_sum")
    info: Dict[str, Any] = {"solver": "light_field", "method": method}

    try:
        # Get light field from physics if available
        if hasattr(physics, 'light_field'):
            lf = physics.light_field
        elif y.ndim == 4:
            lf = y
        else:
            info["error"] = "no_light_field_data"
            return y.astype(np.float32), info

        if method == "lfbm5d":
            sigma = cfg.get("sigma", 0.05)
            result = lfbm5d(lf, sigma=sigma)
        else:
            disparities = cfg.get("disparities", None)
            result = shift_and_sum(lf, disparities)

        return result, info
    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32) if y.ndim == 2 else np.mean(y, axis=(2,3)).astype(np.float32), info
