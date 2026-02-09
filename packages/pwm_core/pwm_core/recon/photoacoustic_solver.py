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


def _precompute_time_indices(
    grid_shape: Tuple[int, int],
    transducer_positions: np.ndarray,
    n_times: int,
    speed_of_sound: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute time indices and validity masks.

    Returns:
        time_indices: (n_trans, ny, nx) integer time bin indices.
        valid_mask: (n_trans, ny, nx) boolean mask for in-range indices.
    """
    dist = _compute_distance_matrix(grid_shape, transducer_positions)
    time_indices = np.round(dist / speed_of_sound).astype(np.int64)
    valid_mask = (time_indices >= 0) & (time_indices < n_times)
    return time_indices, valid_mask


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


def _adjoint_backproject(
    sinogram: np.ndarray,
    transducer_positions: np.ndarray,
    grid_shape: Tuple[int, int],
    speed_of_sound: float = 1.0,
) -> np.ndarray:
    """Adjoint of the forward photoacoustic operator (un-normalized).

    This is the true adjoint A^T satisfying <Ax, y> = <x, A^T y>.
    No normalization is applied so it can be used correctly in iterative
    solvers (CG, Landweber, etc.).

    Args:
        sinogram: (n_transducers, n_times).
        transducer_positions: (n_transducers, 2) positions.
        grid_shape: (ny, nx) reconstruction grid.
        speed_of_sound: Speed of sound.

    Returns:
        Back-projected image (ny, nx) as float64.
    """
    n_trans, n_times = sinogram.shape
    ny, nx = grid_shape

    dist = _compute_distance_matrix(grid_shape, transducer_positions)
    time_indices = np.round(dist / speed_of_sound).astype(np.int64)

    recon = np.zeros((ny, nx), dtype=np.float64)
    for i in range(n_trans):
        ti = time_indices[i]
        valid = (ti >= 0) & (ti < n_times)
        values = np.zeros((ny, nx), dtype=np.float64)
        values[valid] = sinogram[i, ti[valid]]
        recon += values

    return recon


def back_projection(
    sinogram: np.ndarray,
    transducer_positions: np.ndarray,
    grid_shape: Tuple[int, int],
    speed_of_sound: float = 1.0,
) -> np.ndarray:
    """Count-weighted delay-and-sum back-projection for photoacoustic
    reconstruction.

    For each pixel, sums the sinogram values at the appropriate time delay
    from each transducer, then divides by the number of pixels that
    contributed to that sinogram bin (count weighting). This properly
    inverts the many-to-one forward mapping where multiple pixels at the
    same distance from a transducer accumulate into one time bin.

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

    dist = _compute_distance_matrix(grid_shape, transducer_positions)
    time_indices = np.round(dist / speed_of_sound).astype(np.int64)

    # Compute count matrix: how many pixels map to each (transducer, time) bin.
    # sinogram[i, t] = sum of p0 over all pixels at distance t from transducer i,
    # so dividing by count gives the average pixel value on that arc.
    count_matrix = np.zeros((n_trans, n_times), dtype=np.float64)
    for i in range(n_trans):
        ti = time_indices[i]
        valid = (ti >= 0) & (ti < n_times)
        np.add.at(count_matrix[i], ti[valid], 1.0)

    # Back-project with count weighting
    recon = np.zeros((ny, nx), dtype=np.float64)
    for i in range(n_trans):
        ti = time_indices[i]
        valid = (ti >= 0) & (ti < n_times)
        values = np.zeros((ny, nx), dtype=np.float64)
        counts = np.maximum(count_matrix[i, ti[valid]], 1.0)
        values[valid] = sinogram[i, ti[valid]] / counts
        recon += values

    recon /= n_trans

    return np.clip(recon, 0, None).astype(np.float32)


def time_reversal(
    sinogram: np.ndarray,
    transducer_positions: np.ndarray,
    grid_shape: Tuple[int, int],
    speed_of_sound: float = 1.0,
    n_iters: int = 30,
) -> np.ndarray:
    """Iterative photoacoustic reconstruction via Conjugate Gradient on
    Normal Equations (CGNR).

    Solves  min_x ||Ax - b||^2  using CG applied to  A^T A x = A^T b,
    where A is the forward circular Radon transform and A^T is the
    (un-normalized) adjoint back-projection.

    A non-negativity projection is applied after convergence since
    initial pressure p0 >= 0.

    Args:
        sinogram: Measured sinogram (n_transducers, n_times).
        transducer_positions: (n_transducers, 2) positions.
        grid_shape: (ny, nx) reconstruction grid size.
        speed_of_sound: Speed of sound.
        n_iters: Number of CG iterations.

    Returns:
        Reconstructed initial pressure (ny, nx).
    """
    n_trans, n_times = sinogram.shape

    b = sinogram.astype(np.float64)

    # Right-hand side: A^T b
    ATb = _adjoint_backproject(b.astype(np.float32), transducer_positions,
                               grid_shape, speed_of_sound)

    # CG iteration
    x = np.zeros(grid_shape, dtype=np.float64)
    r = ATb.copy()
    p = r.copy()
    rs_old = np.sum(r * r)

    for _ in range(n_iters):
        if rs_old < 1e-20:
            break

        # Compute A^T A p
        Ap = _forward_photoacoustic(p.astype(np.float32), transducer_positions,
                                    n_times, speed_of_sound)
        ATAp = _adjoint_backproject(Ap, transducer_positions,
                                    grid_shape, speed_of_sound)

        pATAp = np.sum(p * ATAp)
        if abs(pATAp) < 1e-20:
            break

        alpha = rs_old / pATAp
        x = x + alpha * p
        r = r - alpha * ATAp
        rs_new = np.sum(r * r)

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    # Non-negativity constraint
    return np.clip(x, 0, None).astype(np.float32)


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
            n_iters = cfg.get("n_iters", 30)
            result = time_reversal(y, trans_pos, grid_shape, sos, n_iters=n_iters)
        else:
            result = back_projection(y, trans_pos, grid_shape, sos)

        return result, info
    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
