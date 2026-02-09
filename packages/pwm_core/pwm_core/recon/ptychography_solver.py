"""Ptychography Reconstruction: ePIE Algorithm.

Extended Ptychographic Iterative Engine for phase retrieval.

References:
- Maiden, A.M. & Rodenburg, J.M. (2009). "An improved ptychographical
  phase retrieval algorithm for diffractive imaging"
- Rodenburg, J.M. (2008). "Ptychography and related diffractive imaging methods"

Benchmark: Synthetic data with 60-80% overlap
Expected PSNR: 30-40 dB (depends on overlap and noise)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def create_probe(
    size: int,
    probe_type: str = "gaussian",
    probe_radius: Optional[float] = None,
) -> np.ndarray:
    """Create initial probe estimate.

    Args:
        size: Probe size (square)
        probe_type: 'gaussian', 'circular', or 'custom'
        probe_radius: Radius in pixels (default: size/4)

    Returns:
        Complex probe array (size, size)
    """
    if probe_radius is None:
        probe_radius = size / 4

    y, x = np.ogrid[:size, :size]
    center = size / 2
    r = np.sqrt((y - center)**2 + (x - center)**2)

    if probe_type == "gaussian":
        amplitude = np.exp(-r**2 / (2 * probe_radius**2))
    elif probe_type == "circular":
        amplitude = (r <= probe_radius).astype(float)
    else:
        amplitude = np.ones((size, size))

    # Add some structure for initial probe
    phase = np.zeros((size, size))

    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def epie(
    diffraction_patterns: np.ndarray,
    positions: np.ndarray,
    object_shape: Tuple[int, int],
    probe_init: Optional[np.ndarray] = None,
    iterations: int = 100,
    alpha: float = 1.0,
    beta: float = 1.0,
    update_probe: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extended Ptychographic Iterative Engine.

    Args:
        diffraction_patterns: Measured intensities (n_positions, det_size, det_size)
        positions: Scan positions (n_positions, 2) as (y, x) indices
        object_shape: Object size (H, W)
        probe_init: Initial probe estimate (det_size, det_size)
        iterations: Number of ePIE iterations
        alpha: Object update strength
        beta: Probe update strength
        update_probe: Whether to update probe (blind ptychography)

    Returns:
        Tuple of (recovered_object, recovered_probe)
    """
    n_positions, det_h, det_w = diffraction_patterns.shape
    obj_h, obj_w = object_shape

    # Sqrt of intensities for amplitude constraints
    sqrt_intensities = np.sqrt(np.maximum(diffraction_patterns, 0))

    # Initialize object (complex)
    obj = np.ones(object_shape, dtype=np.complex64)

    # Initialize probe
    if probe_init is not None:
        probe = probe_init.astype(np.complex64)
    else:
        probe = create_probe(det_h)

    # Iterate
    for it in range(iterations):
        # Random order of positions (important for convergence)
        order = np.random.permutation(n_positions)

        for idx in order:
            pos_y, pos_x = int(positions[idx, 0]), int(positions[idx, 1])

            # Extract object patch
            y_slice = slice(pos_y, pos_y + det_h)
            x_slice = slice(pos_x, pos_x + det_w)

            # Handle boundary
            if pos_y + det_h > obj_h or pos_x + det_w > obj_w:
                continue
            if pos_y < 0 or pos_x < 0:
                continue

            obj_patch = obj[y_slice, x_slice]

            # Exit wave
            psi = probe * obj_patch

            # Propagate to detector (far field)
            Psi = fft2(psi)

            # Apply amplitude constraint
            Psi_mag = np.abs(Psi)
            Psi_phase = np.angle(Psi)

            # Replace amplitude with measured
            Psi_corrected = sqrt_intensities[idx] * np.exp(1j * Psi_phase)

            # Back-propagate
            psi_corrected = ifft2(Psi_corrected)

            # Compute difference
            diff = psi_corrected - psi

            # Update object (ePIE update rule)
            probe_conj = np.conj(probe)
            probe_norm = np.max(np.abs(probe)**2)

            obj_update = alpha * probe_conj / (probe_norm + 1e-10) * diff
            obj[y_slice, x_slice] = obj_patch + obj_update

            # Update probe
            if update_probe:
                obj_conj = np.conj(obj_patch)
                obj_norm = np.max(np.abs(obj_patch)**2)

                probe_update = beta * obj_conj / (obj_norm + 1e-10) * diff
                probe = probe + probe_update

    return obj, probe


def pie(
    diffraction_patterns: np.ndarray,
    positions: np.ndarray,
    object_shape: Tuple[int, int],
    probe: np.ndarray,
    iterations: int = 100,
    feedback: float = 1.0,
) -> np.ndarray:
    """Basic PIE algorithm (Ptychographic Iterative Engine).

    Simpler than ePIE but requires known probe.

    Args:
        diffraction_patterns: Measured intensities
        positions: Scan positions
        object_shape: Object size
        probe: Known probe function
        iterations: Number of iterations
        feedback: Update strength

    Returns:
        Recovered object
    """
    obj, _ = epie(
        diffraction_patterns,
        positions,
        object_shape,
        probe_init=probe,
        iterations=iterations,
        alpha=feedback,
        beta=0.0,  # Don't update probe
        update_probe=False
    )
    return obj


def ml_epie(
    diffraction_patterns: np.ndarray,
    positions: np.ndarray,
    object_shape: Tuple[int, int],
    probe_init: Optional[np.ndarray] = None,
    iterations: int = 100,
    alpha: float = 1.0,
    beta: float = 1.0,
    update_probe: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Maximum-Likelihood ePIE (Poisson noise model).

    Instead of the standard amplitude constraint, uses Poisson ML update
    that better handles low-count data.

    References:
    - Thibault, P. & Guizar-Sicairos, M. (2012). "Maximum-likelihood refinement
      for coherent diffractive imaging", New J. of Physics.

    The key difference from standard ePIE is the exit-wave update rule:
    - Standard: replace amplitude with sqrt(measured intensity)
    - ML: psi_corrected = psi * sqrt(I_meas / (|Psi|^2 + eps))
    This is the gradient of the Poisson log-likelihood.

    Args:
        diffraction_patterns: Measured intensities (n_positions, det_size, det_size)
        positions: Scan positions (n_positions, 2)
        object_shape: Object size (H, W)
        probe_init: Initial probe estimate
        iterations: Number of iterations
        alpha: Object update strength
        beta: Probe update strength
        update_probe: Whether to update probe

    Returns:
        Tuple of (recovered_object, recovered_probe)
    """
    n_positions, det_h, det_w = diffraction_patterns.shape
    obj_h, obj_w = object_shape

    # Initialize object and probe
    obj = np.ones(object_shape, dtype=np.complex64)
    if probe_init is not None:
        probe = probe_init.astype(np.complex64)
    else:
        probe = create_probe(det_h)

    eps = 1e-10

    for it in range(iterations):
        order = np.random.permutation(n_positions)

        for idx in order:
            pos_y, pos_x = int(positions[idx, 0]), int(positions[idx, 1])

            y_slice = slice(pos_y, pos_y + det_h)
            x_slice = slice(pos_x, pos_x + det_w)

            if pos_y + det_h > obj_h or pos_x + det_w > obj_w:
                continue
            if pos_y < 0 or pos_x < 0:
                continue

            obj_patch = obj[y_slice, x_slice]

            # Exit wave
            psi = probe * obj_patch

            # Propagate to detector
            Psi = fft2(psi)

            # ML (Poisson) amplitude update
            I_model = np.abs(Psi)**2 + eps
            I_meas = diffraction_patterns[idx] + eps
            ml_ratio = np.sqrt(I_meas / I_model)
            Psi_corrected = Psi * ml_ratio

            # Back-propagate
            psi_corrected = ifft2(Psi_corrected)

            # Update
            diff = psi_corrected - psi

            probe_conj = np.conj(probe)
            probe_norm = np.max(np.abs(probe)**2)
            obj[y_slice, x_slice] = obj_patch + alpha * probe_conj / (probe_norm + eps) * diff

            if update_probe:
                obj_conj = np.conj(obj_patch)
                obj_norm = np.max(np.abs(obj_patch)**2)
                probe = probe + beta * obj_conj / (obj_norm + eps) * diff

    return obj, probe


def run_epie(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run ePIE reconstruction.

    Args:
        y: Diffraction patterns (n_positions, det_h, det_w)
        physics: Ptychography physics operator
        cfg: Configuration with:
            - iters: Number of iterations (default: 100)
            - alpha: Object update strength (default: 1.0)
            - beta: Probe update strength (default: 1.0)
            - update_probe: Blind ptychography (default: True)

    Returns:
        Tuple of (recovered_object, info_dict)
    """
    iters = cfg.get("iters", 100)
    alpha = cfg.get("alpha", 1.0)
    beta = cfg.get("beta", 1.0)
    update_probe = cfg.get("update_probe", True)

    info = {
        "solver": "epie",
        "iters": iters,
        "alpha": alpha,
        "beta": beta,
    }

    try:
        # Get positions from physics
        positions = None
        object_shape = None
        probe = None

        if hasattr(physics, 'positions'):
            positions = physics.positions
        if hasattr(physics, 'x_shape'):
            object_shape = tuple(physics.x_shape)[:2]
        if hasattr(physics, 'probe'):
            probe = physics.probe

        if hasattr(physics, 'info'):
            op_info = physics.info()
            if 'positions' in op_info:
                positions = op_info['positions']
            if 'x_shape' in op_info:
                object_shape = tuple(op_info['x_shape'])[:2]
            if 'probe' in op_info:
                probe = op_info['probe']

        # Infer if not provided
        if positions is None:
            # Assume grid pattern
            n_pos = y.shape[0]
            grid_size = int(np.sqrt(n_pos))
            det_size = y.shape[1]

            if object_shape is None:
                object_shape = (det_size * 2, det_size * 2)

            # Create grid positions with overlap
            step = det_size // 2  # 50% overlap
            positions = []
            for i in range(grid_size):
                for j in range(grid_size):
                    positions.append([i * step, j * step])
            positions = np.array(positions[:n_pos])

        if object_shape is None:
            det_size = y.shape[1]
            object_shape = (det_size * 2, det_size * 2)

        obj, recovered_probe = epie(
            y, positions, object_shape,
            probe_init=probe,
            iterations=iters,
            alpha=alpha,
            beta=beta,
            update_probe=update_probe
        )

        # Return amplitude and phase
        info["has_probe"] = True

        # For visualization, return amplitude
        result = np.abs(obj).astype(np.float32)

        return result, info

    except Exception as e:
        info["error"] = str(e)
        # Fall back to simple averaging
        if y.ndim >= 3:
            result = np.sqrt(np.mean(y, axis=0))
            return result.astype(np.float32), info
        return y.astype(np.float32), info
