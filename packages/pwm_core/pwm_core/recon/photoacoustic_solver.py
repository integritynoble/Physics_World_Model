"""Photoacoustic reconstruction solvers.

References:
- Xu, M. & Wang, L.V. (2005). "Universal back-projection algorithm for
  photoacoustic computed tomography", Physical Review E.
- Treeby, B.E. & Cox, B.T. (2010). "k-Wave: MATLAB toolbox for the simulation
  and reconstruction of photoacoustic wave fields", J. Biomed. Opt.

Expected PSNR: 32.0 dB on synthetic benchmark
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple


def _compute_distance_matrix(
    grid_shape: Tuple[int, int],
    transducer_positions: np.ndarray,
) -> np.ndarray:
    """Compute distance matrix between all pixels and transducers.

    Args:
        grid_shape: (ny, nx) image grid size.
        transducer_positions: (n_transducers, 2) array of (y, x) positions.

    Returns:
        Distance matrix (n_transducers, ny, nx).
    """
    ny, nx = grid_shape
    yy, xx = np.mgrid[:ny, :nx]  # (ny, nx) each

    # Broadcasting: (n_trans, 1, 1) - (ny, nx) -> (n_trans, ny, nx)
    dy = transducer_positions[:, 0, np.newaxis, np.newaxis] - yy[np.newaxis, :, :]
    dx = transducer_positions[:, 1, np.newaxis, np.newaxis] - xx[np.newaxis, :, :]

    dist = np.sqrt(dy ** 2 + dx ** 2)
    return dist


def _forward_photoacoustic(
    p0: np.ndarray,
    transducer_positions: np.ndarray,
    n_times: int,
    speed_of_sound: float = 1.0,
) -> np.ndarray:
    """Forward photoacoustic model: circular Radon transform.

    sinogram[i, t] = sum_r p0(r) * delta(|r - r_i| - c*t)

    Args:
        p0: Initial pressure (ny, nx).
        transducer_positions: (n_transducers, 2) positions.
        n_times: Number of time samples.
        speed_of_sound: Speed of sound in pixel units per time step.

    Returns:
        Sinogram (n_transducers, n_times).
    """
    ny, nx = p0.shape
    n_trans = len(transducer_positions)

    # Compute distances
    dist = _compute_distance_matrix((ny, nx), transducer_positions)

    # Convert distances to time indices
    time_indices = np.round(dist / speed_of_sound).astype(np.int64)

    # Build sinogram using vectorized accumulation
    sinogram = np.zeros((n_trans, n_times), dtype=np.float64)

    for i in range(n_trans):
        ti = time_indices[i]  # (ny, nx)
        valid = (ti >= 0) & (ti < n_times)
        np.add.at(sinogram[i], ti[valid], p0[valid].astype(np.float64))

    return sinogram.astype(np.float32)


def back_projection(
    sinogram: np.ndarray,
    transducer_positions: np.ndarray,
    grid_shape: Tuple[int, int],
    speed_of_sound: float = 1.0,
) -> np.ndarray:
    """Delay-and-sum back-projection for photoacoustic reconstruction.

    For each pixel, sums the sinogram values at the appropriate time
    delay from each transducer.

    Args:
        sinogram: Measured sinogram (n_transducers, n_times).
        transducer_positions: (n_transducers, 2) positions.
        grid_shape: (ny, nx) reconstruction grid size.
        speed_of_sound: Speed of sound.

    Returns:
        Reconstructed initial pressure (ny, nx).
    """
    n_trans, n_times = sinogram.shape
    ny, nx = grid_shape

    # Compute distance matrix
    dist = _compute_distance_matrix(grid_shape, transducer_positions)

    # Convert to time indices
    time_indices = np.round(dist / speed_of_sound).astype(np.int64)

    # Back-project: sum sinogram values at computed delays
    recon = np.zeros((ny, nx), dtype=np.float64)

    for i in range(n_trans):
        ti = time_indices[i]  # (ny, nx)
        valid = (ti >= 0) & (ti < n_times)
        # Gather values from sinogram
        values = np.zeros((ny, nx), dtype=np.float64)
        values[valid] = sinogram[i, ti[valid]]
        recon += values

    # Normalize by number of transducers
    recon /= n_trans

    return np.clip(recon, 0, None).astype(np.float32)


def time_reversal(
    sinogram: np.ndarray,
    transducer_positions: np.ndarray,
    grid_shape: Tuple[int, int],
    speed_of_sound: float = 1.0,
    n_iters: int = 20,
) -> np.ndarray:
    """Iterative time-reversal photoacoustic reconstruction.

    Refines the back-projection estimate by iterating forward-backward
    projections.

    Args:
        sinogram: Measured sinogram (n_transducers, n_times).
        transducer_positions: (n_transducers, 2) positions.
        grid_shape: (ny, nx) reconstruction grid size.
        speed_of_sound: Speed of sound.
        n_iters: Number of refinement iterations.

    Returns:
        Reconstructed initial pressure (ny, nx).
    """
    n_trans, n_times = sinogram.shape

    # Initial estimate via back-projection
    recon = back_projection(sinogram, transducer_positions, grid_shape, speed_of_sound)

    step_size = 0.5

    for it in range(n_iters):
        # Forward project current estimate
        sino_est = _forward_photoacoustic(recon, transducer_positions,
                                          n_times, speed_of_sound)

        # Compute residual sinogram
        residual = sinogram - sino_est

        # Back-project residual
        correction = back_projection(residual, transducer_positions,
                                     grid_shape, speed_of_sound)

        # Update
        recon = recon + step_size * correction

        # Non-negativity constraint
        recon = np.clip(recon, 0, None)

    return recon.astype(np.float32)


def run_photoacoustic(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for photoacoustic reconstruction.

    Args:
        y: Sinogram (n_transducers, n_times).
        physics: Physics operator with transducer_positions and grid_shape.
        cfg: Configuration dict.

    Returns:
        Tuple of (reconstructed_pressure, info_dict).
    """
    method = cfg.get("method", "back_projection")
    info: Dict[str, Any] = {"solver": "photoacoustic", "method": method}

    try:
        trans_pos = getattr(physics, 'transducer_positions', None)
        grid_shape = getattr(physics, 'grid_shape', (128, 128))
        sos = getattr(physics, 'speed_of_sound', 1.0)

        if trans_pos is None:
            info["error"] = "no_transducer_positions"
            return y.astype(np.float32), info

        if method == "time_reversal":
            n_iters = cfg.get("n_iters", 20)
            result = time_reversal(y, trans_pos, grid_shape, sos, n_iters=n_iters)
        else:
            result = back_projection(y, trans_pos, grid_shape, sos)

        return result, info
    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
