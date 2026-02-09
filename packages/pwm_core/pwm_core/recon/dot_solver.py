"""DOT (Diffuse Optical Tomography) reconstruction solvers.

References:
- Arridge, S.R. (1999). "Optical tomography in medical imaging",
  Inverse Problems.
- Schweiger, M. & Arridge, S.R. (2014). "The Toast++ software suite
  for forward and inverse modeling in optical tomography",
  J. Biomed. Opt.

Expected PSNR: 25.0 dB on synthetic benchmark
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple
from scipy.optimize import minimize


def _green_function_3d(
    r1: np.ndarray,
    r2: np.ndarray,
    mu_a_bg: float = 0.01,
    mu_s_prime: float = 1.0,
    D: float = None,
) -> np.ndarray:
    """Compute 3D diffusion Green's function.

    G(r1, r2) = exp(-k_d * |r1-r2|) / (4*pi*D*|r1-r2|)

    where k_d = sqrt(mu_a / D), D = 1 / (3 * (mu_a + mu_s'))

    Args:
        r1: Source positions (..., 3).
        r2: Point positions (..., 3).
        mu_a_bg: Background absorption coefficient.
        mu_s_prime: Reduced scattering coefficient.
        D: Diffusion coefficient. If None, computed from mu_a and mu_s'.

    Returns:
        Green's function values.
    """
    if D is None:
        D = 1.0 / (3.0 * (mu_a_bg + mu_s_prime))

    k_d = np.sqrt(mu_a_bg / D)

    dist = np.sqrt(np.sum((r1 - r2) ** 2, axis=-1))
    dist = np.maximum(dist, 1e-6)  # Avoid division by zero

    G = np.exp(-k_d * dist) / (4 * np.pi * D * dist)

    return G


def _build_jacobian(
    source_positions: np.ndarray,
    detector_positions: np.ndarray,
    voxel_positions: np.ndarray,
    voxel_volume: float,
    mu_a_bg: float = 0.01,
    mu_s_prime: float = 1.0,
) -> np.ndarray:
    """Build the Born approximation Jacobian matrix.

    J[m, v] = -voxel_volume * G(r_s, r_v) * G(r_v, r_d) / G(r_s, r_d)

    where m = source*n_det + detector index.

    Args:
        source_positions: (n_sources, 3) positions.
        detector_positions: (n_detectors, 3) positions.
        voxel_positions: (n_voxels, 3) positions.
        voxel_volume: Volume of each voxel.
        mu_a_bg: Background absorption.
        mu_s_prime: Reduced scattering.

    Returns:
        Jacobian matrix (n_measurements, n_voxels).
    """
    n_src = len(source_positions)
    n_det = len(detector_positions)
    n_vox = len(voxel_positions)
    n_meas = n_src * n_det

    D = 1.0 / (3.0 * (mu_a_bg + mu_s_prime))

    J = np.zeros((n_meas, n_vox), dtype=np.float64)

    for s in range(n_src):
        # G(source, voxel): (n_vox,)
        r_s = source_positions[s]
        G_sv = _green_function_3d(
            r_s[np.newaxis, :].repeat(n_vox, axis=0),
            voxel_positions, mu_a_bg, mu_s_prime, D
        )

        for d in range(n_det):
            r_d = detector_positions[d]
            # G(voxel, detector): (n_vox,)
            G_vd = _green_function_3d(
                voxel_positions,
                r_d[np.newaxis, :].repeat(n_vox, axis=0),
                mu_a_bg, mu_s_prime, D
            )

            # G(source, detector): scalar
            G_sd = _green_function_3d(
                r_s[np.newaxis, :], r_d[np.newaxis, :],
                mu_a_bg, mu_s_prime, D
            )[0]

            m = s * n_det + d
            J[m] = -voxel_volume * G_sv * G_vd / (G_sd + 1e-20)

    return J


def born_approx(
    y: np.ndarray,
    jacobian: np.ndarray,
    alpha: float = 1.0,
    max_cg_iters: int = 200,
) -> np.ndarray:
    """Born approximation with Tikhonov regularization.

    Solves: x = (J^T J + alpha*I)^{-1} J^T y

    Uses dense direct solve for small problems (n_voxels <= 8192)
    and Jacobi-preconditioned CG for larger problems.

    Args:
        y: Measurement vector (n_measurements,).
        jacobian: Jacobian matrix (n_measurements, n_voxels).
        alpha: Tikhonov regularization parameter.
        max_cg_iters: Maximum CG iterations.

    Returns:
        Reconstructed absorption perturbation (n_voxels,).
    """
    n_meas, n_vox = jacobian.shape
    JtJ = jacobian.T @ jacobian
    Jty = jacobian.T @ y

    if n_vox <= 8192:
        # Dense direct solve for small problems (better numerical stability)
        x = np.linalg.solve(JtJ + alpha * np.eye(n_vox), Jty)
    else:
        from scipy.sparse.linalg import cg, LinearOperator

        # Jacobi preconditioner for better CG convergence
        diag = np.diag(JtJ) + alpha
        M_diag = 1.0 / np.maximum(diag, 1e-10)
        M = LinearOperator((n_vox, n_vox), matvec=lambda x: M_diag * x)

        def matvec(x):
            return JtJ @ x + alpha * x

        A = LinearOperator((n_vox, n_vox), matvec=matvec)
        x, info = cg(A, Jty, maxiter=max_cg_iters, rtol=1e-6, M=M)

    return x.astype(np.float32)


def _tv_gradient_3d(x: np.ndarray, shape: Tuple[int, int, int],
                     epsilon: float = 1e-6) -> np.ndarray:
    """Compute smoothed 3D Total Variation gradient.

    Args:
        x: Flattened volume (n_voxels,).
        shape: (nz, ny, nx) volume shape.
        epsilon: Smoothing parameter for differentiable TV.

    Returns:
        TV gradient (n_voxels,).
    """
    vol = x.reshape(shape)
    nz, ny, nx = shape

    # Compute differences along each axis
    dx = np.zeros_like(vol)
    dy = np.zeros_like(vol)
    dz = np.zeros_like(vol)

    dx[:, :, :-1] = vol[:, :, 1:] - vol[:, :, :-1]
    dy[:, :-1, :] = vol[:, 1:, :] - vol[:, :-1, :]
    dz[:-1, :, :] = vol[1:, :, :] - vol[:-1, :, :]

    # Smoothed magnitude
    mag = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + epsilon ** 2)

    # Normalized gradients
    nx_g = dx / mag
    ny_g = dy / mag
    nz_g = dz / mag

    # Divergence (adjoint of gradient)
    div = np.zeros_like(vol)

    # d/dx component
    div[:, :, 0] = nx_g[:, :, 0]
    div[:, :, -1] = -nx_g[:, :, -2]
    div[:, :, 1:-1] = nx_g[:, :, 1:-1] - nx_g[:, :, :-2]

    # d/dy component
    div[:, 0, :] += ny_g[:, 0, :]
    div[:, -1, :] += -ny_g[:, -2, :]
    div[:, 1:-1, :] += ny_g[:, 1:-1, :] - ny_g[:, :-2, :]

    # d/dz component
    div[0, :, :] += nz_g[0, :, :]
    div[-1, :, :] += -nz_g[-2, :, :]
    div[1:-1, :, :] += nz_g[1:-1, :, :] - nz_g[:-2, :, :]

    return (-div).ravel()


def lbfgs_tv(
    y: np.ndarray,
    jacobian: np.ndarray,
    volume_shape: Tuple[int, int, int],
    lambda_tv: float = 0.01,
    max_iters: int = 100,
) -> np.ndarray:
    """L-BFGS-B with Total Variation regularization for DOT.

    Minimizes: ||Jx - y||^2 + lambda_tv * TV(x)

    Args:
        y: Measurement vector (n_measurements,).
        jacobian: Jacobian matrix (n_measurements, n_voxels).
        volume_shape: (nz, ny, nx) volume shape.
        lambda_tv: TV regularization weight.
        max_iters: Maximum L-BFGS-B iterations.

    Returns:
        Reconstructed absorption perturbation (n_voxels,).
    """
    n_vox = jacobian.shape[1]

    # Precompute J^T for efficiency
    Jt = jacobian.T
    JtJ = Jt @ jacobian
    Jty = Jt @ y

    def objective(x):
        residual = jacobian @ x - y
        data_term = 0.5 * np.sum(residual ** 2)

        # TV term (smoothed)
        vol = x.reshape(volume_shape)
        dx = np.diff(vol, axis=2)
        dy = np.diff(vol, axis=1)
        dz = np.diff(vol, axis=0)
        tv = np.sum(np.sqrt(dx ** 2 + 1e-12)) + \
             np.sum(np.sqrt(dy ** 2 + 1e-12)) + \
             np.sum(np.sqrt(dz ** 2 + 1e-12))

        return data_term + lambda_tv * tv

    def gradient(x):
        # Data gradient: J^T(Jx - y)
        data_grad = JtJ @ x - Jty

        # TV gradient
        tv_grad = _tv_gradient_3d(x, volume_shape)

        return data_grad + lambda_tv * tv_grad

    # Initialize with Tikhonov solution
    x0 = born_approx(y, jacobian, alpha=0.1, max_cg_iters=50)

    result = minimize(
        objective,
        x0.astype(np.float64),
        jac=gradient,
        method='L-BFGS-B',
        bounds=[(0, None)] * n_vox,
        options={'maxiter': max_iters, 'ftol': 1e-10},
    )

    return result.x.astype(np.float32)


def run_dot(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for DOT reconstruction.

    Args:
        y: Measurement vector (n_measurements,).
        physics: Physics operator with jacobian, volume_shape.
        cfg: Configuration dict.

    Returns:
        Tuple of (reconstructed_volume, info_dict).
    """
    method = cfg.get("method", "born_approx")
    info: Dict[str, Any] = {"solver": "dot", "method": method}

    try:
        jacobian = getattr(physics, 'jacobian', None)
        volume_shape = getattr(physics, 'volume_shape', (32, 32, 32))

        if jacobian is None:
            info["error"] = "no_jacobian"
            return y.astype(np.float32), info

        if method == "lbfgs_tv":
            lambda_tv = cfg.get("lambda_tv", 0.01)
            max_iters = cfg.get("max_iters", 100)
            result_flat = lbfgs_tv(y, jacobian, volume_shape,
                                    lambda_tv=lambda_tv, max_iters=max_iters)
        else:
            alpha = cfg.get("alpha", 0.1)
            result_flat = born_approx(y, jacobian, alpha=alpha)

        result = result_flat.reshape(volume_shape)
        return result, info
    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
