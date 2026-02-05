"""pwm_core.recon.pnp

Plug-and-Play (PnP) reconstruction with deep denoisers.

Implements PnP-ADMM and PnP-HQS algorithms with:
- DnCNN / DRUNet denoisers (if available)
- BM3D fallback
- Gaussian denoiser fallback

References:
- Venkatakrishnan et al., "Plug-and-Play Priors for Model Based Reconstruction"
- Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior"
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

# Try to import deep learning denoisers
_HAS_TORCH = False
_HAS_DRUNET = False
_HAS_BM3D = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

try:
    import bm3d
    _HAS_BM3D = True
except ImportError:
    pass


def gaussian_denoiser(x: np.ndarray, sigma: float) -> np.ndarray:
    """Simple Gaussian filter denoiser (fallback)."""
    from scipy.ndimage import gaussian_filter
    # Sigma for filter scales with noise level
    filter_sigma = max(0.5, sigma * 5)
    return gaussian_filter(x, sigma=filter_sigma).astype(np.float32)


def nlm_denoiser(x: np.ndarray, sigma: float) -> np.ndarray:
    """Non-local means denoiser."""
    try:
        from skimage.restoration import denoise_nl_means, estimate_sigma
        # Estimate patch/search parameters based on noise level
        patch_kw = dict(
            patch_size=5,
            patch_distance=6,
            h=0.8 * sigma,
            fast_mode=True,
            sigma=sigma,
        )
        return denoise_nl_means(x, **patch_kw).astype(np.float32)
    except ImportError:
        return gaussian_denoiser(x, sigma)


def bm3d_denoiser(x: np.ndarray, sigma: float) -> np.ndarray:
    """BM3D denoiser."""
    if not _HAS_BM3D:
        return nlm_denoiser(x, sigma)

    # Normalize to [0, 1] range for BM3D
    x_min, x_max = x.min(), x.max()
    if x_max - x_min > 1e-8:
        x_norm = (x - x_min) / (x_max - x_min)
    else:
        x_norm = x - x_min

    # BM3D expects sigma in [0, 1] range
    sigma_norm = sigma / max(x_max - x_min, 1.0)

    try:
        denoised = bm3d.bm3d(x_norm, sigma_psd=sigma_norm, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        # Denormalize
        return (denoised * (x_max - x_min) + x_min).astype(np.float32)
    except Exception:
        return nlm_denoiser(x, sigma)


class DRUNetDenoiser:
    """DRUNet deep denoiser wrapper."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load pre-trained DRUNet model."""
        if not _HAS_TORCH:
            return

        try:
            # Try to load from deepinv
            from deepinv.models import DRUNet
            self.model = DRUNet(in_channels=1, out_channels=1, pretrained="download")
            self.model.to(self.device)
            self.model.eval()
        except ImportError:
            try:
                # Try loading from local weights
                from pwm_core.recon.drunet_arch import DRUNet
                self.model = DRUNet(in_nc=1, out_nc=1, nc=[64, 128, 256, 512])
                # weights_path = Path(__file__).parent / "weights" / "drunet_gray.pth"
                # if weights_path.exists():
                #     self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
            except Exception:
                self.model = None

    def __call__(self, x: np.ndarray, sigma: float) -> np.ndarray:
        if self.model is None or not _HAS_TORCH:
            return bm3d_denoiser(x, sigma)

        import torch

        # Handle different input shapes
        original_shape = x.shape
        if x.ndim == 2:
            x_tensor = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            # For 3D, process slice by slice
            result = np.zeros_like(x)
            for i in range(x.shape[2]):
                result[:, :, i] = self(x[:, :, i], sigma)
            return result
        else:
            return bm3d_denoiser(x, sigma)

        x_tensor = x_tensor.to(self.device)

        # Normalize
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-8:
            x_tensor = (x_tensor - x_min) / (x_max - x_min)

        # Add noise level map for DRUNet
        sigma_norm = sigma / max(x_max - x_min, 1.0)
        sigma_map = torch.ones_like(x_tensor) * sigma_norm

        with torch.no_grad():
            try:
                # DRUNet from deepinv
                denoised = self.model(x_tensor, sigma_map)
            except TypeError:
                # Standard DRUNet
                x_with_sigma = torch.cat([x_tensor, sigma_map], dim=1)
                denoised = self.model(x_with_sigma)

        # Denormalize
        denoised = denoised * (x_max - x_min) + x_min
        result = denoised.squeeze().cpu().numpy().astype(np.float32)

        return result.reshape(original_shape[:2])


def get_denoiser(denoiser_type: str = "auto", device: str = "cpu") -> Callable:
    """Get denoiser function based on type.

    Args:
        denoiser_type: One of 'drunet', 'bm3d', 'nlm', 'gaussian', 'auto'
        device: Device for deep learning denoisers

    Returns:
        Denoiser callable (x, sigma) -> denoised_x
    """
    if denoiser_type == "auto":
        # Try best available
        if _HAS_TORCH:
            try:
                denoiser = DRUNetDenoiser(device)
                if denoiser.model is not None:
                    return denoiser
            except Exception:
                pass
        if _HAS_BM3D:
            return bm3d_denoiser
        return nlm_denoiser

    elif denoiser_type == "drunet":
        return DRUNetDenoiser(device)
    elif denoiser_type == "bm3d":
        return bm3d_denoiser
    elif denoiser_type == "nlm":
        return nlm_denoiser
    elif denoiser_type == "gaussian":
        return gaussian_denoiser
    else:
        return nlm_denoiser


def pnp_admm(
    y: np.ndarray,
    forward: Callable,
    adjoint: Callable,
    x_shape: Tuple[int, ...],
    denoiser: Callable,
    iters: int = 50,
    rho: float = 1.0,
    sigma: float = 0.1,
    sigma_decay: float = 0.95,
) -> np.ndarray:
    """PnP-ADMM reconstruction.

    Solves: min_x 0.5||A(x) - y||^2 + R(x)
    where R(x) is implicitly defined by the denoiser.

    ADMM iterations:
        x^{k+1} = (A^T A + rho I)^{-1} (A^T y + rho(z^k - u^k))
        z^{k+1} = D_sigma(x^{k+1} + u^k)
        u^{k+1} = u^k + x^{k+1} - z^{k+1}

    Args:
        y: Measurements
        forward: Forward operator A
        adjoint: Adjoint operator A^T
        x_shape: Shape of x
        denoiser: Denoiser function D_sigma(x, sigma)
        iters: Number of iterations
        rho: ADMM penalty parameter
        sigma: Initial denoiser noise level
        sigma_decay: Decay factor for sigma per iteration

    Returns:
        Reconstructed x
    """
    # Initialize
    x = adjoint(y).reshape(x_shape).astype(np.float32)
    z = x.copy()
    u = np.zeros_like(x)

    # Precompute A^T y
    Aty = adjoint(y).reshape(x_shape).astype(np.float32)

    current_sigma = sigma

    for k in range(iters):
        # x-update: solve (A^T A + rho I) x = A^T y + rho(z - u)
        # Use conjugate gradient for this
        rhs = Aty + rho * (z - u)

        # Simple iterative solver for x-update
        x_new = x.copy()
        for _ in range(10):
            AtAx = adjoint(forward(x_new)).reshape(x_shape)
            grad = AtAx + rho * x_new - rhs
            x_new = x_new - 0.1 * grad
        x = x_new

        # z-update: denoising
        v = x + u
        z = denoiser(v, current_sigma)

        # Ensure z has correct shape
        if z.shape != x_shape:
            z = z.reshape(x_shape[:2]) if z.ndim == 2 else z
            if z.shape != x_shape:
                z = np.broadcast_to(z[..., np.newaxis] if z.ndim == 2 else z, x_shape).copy()

        # u-update
        u = u + x - z

        # Decay sigma
        current_sigma *= sigma_decay

    return x.astype(np.float32)


def pnp_hqs(
    y: np.ndarray,
    forward: Callable,
    adjoint: Callable,
    x_shape: Tuple[int, ...],
    denoiser: Callable,
    iters: int = 30,
    rho: float = 1.0,
    sigma: float = 0.1,
    sigma_decay: float = 0.9,
) -> np.ndarray:
    """PnP-HQS (Half Quadratic Splitting) reconstruction.

    Simpler than ADMM, often works better for imaging.

    HQS iterations:
        x^{k+1} = (A^T A + rho I)^{-1} (A^T y + rho * z^k)
        z^{k+1} = D_sigma(x^{k+1})

    Args:
        y: Measurements
        forward: Forward operator A
        adjoint: Adjoint operator A^T
        x_shape: Shape of x
        denoiser: Denoiser function D_sigma(x, sigma)
        iters: Number of iterations
        rho: Penalty parameter (higher = more weight on denoiser)
        sigma: Initial denoiser noise level
        sigma_decay: Decay factor for sigma

    Returns:
        Reconstructed x
    """
    # Initialize with adjoint
    z = adjoint(y).reshape(x_shape).astype(np.float32)

    # Normalize z to reasonable range [0, 1]
    z_min = z.min()
    z_max = z.max()
    z_range = z_max - z_min
    if z_range > 1e-8:
        z = (z - z_min) / z_range
    else:
        z = z - z_min

    # Also normalize y for consistency
    y_norm = y.copy()

    Aty = adjoint(y).reshape(x_shape).astype(np.float32)
    if z_range > 1e-8:
        Aty = (Aty - z_min) / z_range

    current_sigma = sigma

    for k in range(iters):
        # x-update: gradient steps for (A^T A + rho I) x = A^T y + rho * z
        x = z.copy()
        rhs = Aty + rho * z

        # Adaptive step size
        step = 0.1 / (1 + rho + k * 0.1)

        for inner in range(10):
            # Compute A^T A x (denormalize, forward, normalize back)
            x_denorm = x * z_range + z_min if z_range > 1e-8 else x + z_min
            AtAx = adjoint(forward(x_denorm)).reshape(x_shape)
            if z_range > 1e-8:
                AtAx = (AtAx - z_min) / z_range

            grad = AtAx + rho * x - rhs
            x = x - step * grad

            # Project to valid range
            x = np.clip(x, 0, 1)

        # z-update: denoising
        # Handle different dimensions
        if x.ndim == 2:
            z = denoiser(x, current_sigma)
        elif x.ndim == 3:
            # For 3D data, denoise slice by slice
            z = np.zeros_like(x)
            for i in range(x.shape[2]):
                z[:, :, i] = denoiser(x[:, :, i], current_sigma)
        else:
            z = x  # Fallback

        # Ensure z has correct shape and range
        z = np.clip(z.reshape(x_shape), 0, 1)

        # Decay sigma (noise level decreases as we iterate)
        current_sigma = max(current_sigma * sigma_decay, 0.01)

    # Denormalize
    z = z * z_range + z_min if z_range > 1e-8 else z + z_min

    return z.astype(np.float32)


def pnp_fista(
    y: np.ndarray,
    forward: Callable,
    adjoint: Callable,
    x_shape: Tuple[int, ...],
    denoiser: Callable,
    iters: int = 50,
    step: float = 0.01,
    sigma: float = 0.1,
    sigma_decay: float = 0.95,
) -> np.ndarray:
    """PnP-FISTA reconstruction.

    FISTA with denoiser as proximal operator.

    Args:
        y: Measurements
        forward: Forward operator A
        adjoint: Adjoint operator A^T
        x_shape: Shape of x
        denoiser: Denoiser function
        iters: Number of iterations
        step: Step size for gradient
        sigma: Initial denoiser noise level
        sigma_decay: Decay factor for sigma

    Returns:
        Reconstructed x
    """
    # Initialize
    x = adjoint(y).reshape(x_shape).astype(np.float32)
    z = x.copy()
    t = 1.0

    # Normalize
    x_max = np.abs(x).max()
    if x_max > 0:
        x = x / x_max
        z = z / x_max

    current_sigma = sigma

    for k in range(iters):
        # Gradient step
        residual = forward(z * x_max if x_max > 0 else z) - y
        grad = adjoint(residual).reshape(x_shape)
        if x_max > 0:
            grad = grad / x_max

        # Gradient descent
        v = z - step * grad

        # Denoiser as proximal operator
        if v.ndim == 2:
            x_new = denoiser(v, current_sigma * step)
        elif v.ndim == 3:
            x_new = np.zeros_like(v)
            for i in range(v.shape[2]):
                x_new[:, :, i] = denoiser(v[:, :, i], current_sigma * step)
        else:
            x_new = v

        x_new = x_new.reshape(x_shape)

        # FISTA momentum
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x = x_new
        t = t_new

        # Decay sigma
        current_sigma *= sigma_decay

    # Denormalize
    if x_max > 0:
        x = x * x_max

    return x.astype(np.float32)


def run_pnp(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run PnP reconstruction.

    Args:
        y: Measurements
        physics: Physics operator with forward/adjoint
        cfg: Configuration with:
            - algorithm: 'admm', 'hqs', or 'fista' (default: 'hqs')
            - denoiser: 'auto', 'drunet', 'bm3d', 'nlm', 'gaussian'
            - iters: Number of iterations (default: 30)
            - sigma: Initial noise level (default: 0.1)
            - rho: Penalty parameter (default: 1.0)

    Returns:
        Tuple of (reconstructed_x, info_dict)
    """
    if not (hasattr(physics, 'forward') and hasattr(physics, 'adjoint')):
        return y.astype(np.float32), {"error": "operator has no forward/adjoint"}

    # Get x_shape
    x_shape = None
    if hasattr(physics, 'x_shape'):
        x_shape = tuple(physics.x_shape)
    elif hasattr(physics, 'info'):
        info = physics.info()
        if 'x_shape' in info:
            x_shape = tuple(info['x_shape'])

    if x_shape is None:
        x_shape = y.shape

    # Get denoiser
    denoiser_type = cfg.get("denoiser", "auto")
    device = cfg.get("device", "cpu")
    denoiser = get_denoiser(denoiser_type, device)

    # Get algorithm parameters
    algorithm = cfg.get("algorithm", "hqs")
    iters = cfg.get("iters", 30)
    sigma = cfg.get("sigma", 0.1)
    sigma_decay = cfg.get("sigma_decay", 0.9)
    rho = cfg.get("rho", 1.0)

    try:
        if algorithm == "admm":
            x = pnp_admm(
                y=y,
                forward=physics.forward,
                adjoint=physics.adjoint,
                x_shape=x_shape,
                denoiser=denoiser,
                iters=iters,
                rho=rho,
                sigma=sigma,
                sigma_decay=sigma_decay,
            )
        elif algorithm == "fista":
            step = cfg.get("step", 0.01)
            x = pnp_fista(
                y=y,
                forward=physics.forward,
                adjoint=physics.adjoint,
                x_shape=x_shape,
                denoiser=denoiser,
                iters=iters,
                step=step,
                sigma=sigma,
                sigma_decay=sigma_decay,
            )
        else:  # hqs (default)
            x = pnp_hqs(
                y=y,
                forward=physics.forward,
                adjoint=physics.adjoint,
                x_shape=x_shape,
                denoiser=denoiser,
                iters=iters,
                rho=rho,
                sigma=sigma,
                sigma_decay=sigma_decay,
            )

        info = {
            "solver": f"pnp_{algorithm}",
            "denoiser": denoiser_type,
            "iters": iters,
            "sigma": sigma,
            "rho": rho,
        }
        return x, info

    except Exception as e:
        return y.astype(np.float32), {"error": str(e)}
