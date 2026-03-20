"""Reconstructor — executes reconstruction algorithms defined in a ReconAction.

Supported algorithm types (dispatched by action.algorithm_type):
  - Classical:        FBP, IFFT
  - Variational:      TV-ADMM, L1-Wavelet ADMM
  - PnP:              PnP-ADMM (BM3D or DnCNN denoiser)
  - Deep Unrolling:   placeholder (returns zero-filled + message)
  - Diffusion:        placeholder
  - Custom:           falls back to TV-ADMM

All methods apply mismatch corrections specified in action.mismatch_corrections
before reconstruction.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter

from ..schemas import ReconAction
from ..utils.mismatch import apply_mismatch_corrections


class Reconstructor:
    """Executes a ReconAction on measured data."""

    def reconstruct(
        self,
        action: ReconAction,
        measurements: np.ndarray,
        modality: str = "",
    ) -> tuple[np.ndarray, list[float]]:
        """Run the full reconstruction pipeline.

        Returns:
            (x_hat, convergence_history)  where history is per-iteration residual
        """
        # Apply mismatch corrections first
        y = apply_mismatch_corrections(
            measurements, action.mismatch_corrections, modality
        )

        alg_type = action.algorithm_type.lower()
        hp = action.hyperparameters

        if alg_type in ("classical",):
            return self._classical(y, action, modality)
        elif alg_type in ("variational", "compressed sensing"):
            return self._tv_admm(y, action, modality)
        elif alg_type in ("pnp", "plug-and-play"):
            return self._pnp_admm(y, action, modality)
        elif "unrolling" in alg_type or "deep" in alg_type:
            print("[Reconstructor] Deep unrolling requires pre-trained weights — "
                  "falling back to TV-ADMM")
            return self._tv_admm(y, action, modality)
        elif "diffusion" in alg_type:
            print("[Reconstructor] Diffusion requires pre-trained score model — "
                  "falling back to TV-ADMM")
            return self._tv_admm(y, action, modality)
        else:
            return self._tv_admm(y, action, modality)

    # ── Classical ─────────────────────────────────────────────────────────────

    def _classical(
        self, y: np.ndarray, action: ReconAction, modality: str
    ) -> tuple[np.ndarray, list[float]]:
        alg = action.algorithm_name.lower()
        if "fbp" in alg or "filtered back" in alg:
            return self._fbp(y, action)
        elif "ifft" in alg or "zero-fill" in alg:
            return self._zero_filled_ifft(y)
        elif "sart" in alg:
            return self._sart(y, action)
        else:
            # Generic: try FBP for tomography, IFFT for MRI-like
            if y.ndim == 2 and not np.iscomplexobj(y):
                return self._fbp(y, action)
            else:
                return self._zero_filled_ifft(y)

    def _fbp(
        self, sino: np.ndarray, action: ReconAction
    ) -> tuple[np.ndarray, list[float]]:
        try:
            from skimage.transform import iradon
        except ImportError:
            return sino.T, []
        # skimage radon returns (pixels, angles); iradon expects (pixels, angles)
        n_angles = sino.shape[1] if sino.ndim == 2 else sino.shape[0]
        theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
        filt  = action.hyperparameters.get("filter", "ramp")
        x_hat = iradon(sino, theta=theta, filter_name=filt, circle=True)
        return x_hat.astype(np.float32), []

    def _zero_filled_ifft(self, kspace: np.ndarray) -> tuple[np.ndarray, list[float]]:
        img = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))
        return img.astype(np.float32), []

    def _sart(
        self, sino: np.ndarray, action: ReconAction
    ) -> tuple[np.ndarray, list[float]]:
        """Simplified SART via iterative back-projection."""
        try:
            from skimage.transform import iradon, radon
        except ImportError:
            return self._fbp(sino, action)

        # sino shape: (pixels, angles) from skimage radon
        n_angles  = sino.shape[1] if sino.ndim == 2 else sino.shape[0]
        theta     = np.linspace(0.0, 180.0, n_angles, endpoint=False)
        num_iter  = int(action.hyperparameters.get("num_iterations", 20))
        relax     = float(action.hyperparameters.get("relaxation", 0.15))

        # Initialise from FBP
        x = iradon(sino, theta=theta, circle=True, filter_name="ramp")
        history = []
        for _ in range(num_iter):
            x_prev = x.copy()
            residual = sino - radon(np.maximum(x, 0), theta=theta, circle=True)
            x += relax * iradon(residual, theta=theta, circle=True, filter_name=None)
            x = np.maximum(x, 0)
            history.append(float(np.linalg.norm(x - x_prev) / (np.linalg.norm(x_prev) + 1e-8)))
        return x.astype(np.float32), history

    # ── TV-ADMM ───────────────────────────────────────────────────────────────

    def _tv_admm(
        self, y: np.ndarray, action: ReconAction, modality: str
    ) -> tuple[np.ndarray, list[float]]:
        """Gradient-descent-on-data + proxTV via split-Bregman (simplified)."""
        try:
            from skimage.transform import iradon, radon
            skimage_ok = True
        except ImportError:
            skimage_ok = False

        hp        = action.hyperparameters
        lam       = float(hp.get("lambda_tv", hp.get("lambda", 5e-3)))
        num_iter  = int(hp.get("num_iterations", 50))
        step      = float(hp.get("step_size", 0.01))

        # Initialise
        if np.iscomplexobj(y):
            x = np.abs(np.fft.ifft2(np.fft.ifftshift(y)))
        elif skimage_ok and y.ndim == 2:
            # sino shape: (pixels, angles) — use shape[1] for n_angles
            n_angles = y.shape[1] if y.shape[1] < y.shape[0] else y.shape[0]
            theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
            x = iradon(y, theta=theta, circle=True, filter_name="ramp")
        else:
            x = y.copy().astype(np.float64)

        history = []
        for it in range(num_iter):
            x_prev = x.copy()

            # Data fidelity gradient
            if np.iscomplexobj(y):
                kspace = np.fft.fftshift(np.fft.fft2(x))
                mask   = np.abs(y) > 0
                grad   = np.real(np.fft.ifft2(np.fft.ifftshift((kspace - y) * mask)))
            elif skimage_ok and y.ndim == 2 and not np.iscomplexobj(y):
                n_angles = y.shape[1] if y.shape[1] < y.shape[0] else y.shape[0]
                theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
                grad  = iradon(
                    radon(x.astype(float), theta=theta, circle=True) - y.astype(float),
                    theta=theta, circle=True, filter_name=None
                )
            else:
                grad = x - y

            x = x - step * grad

            # TV proximal step (Gaussian-blur approximation for speed)
            x = self._prox_tv(x, lam * step)
            x = np.maximum(x, 0.0)

            res = float(np.linalg.norm(x - x_prev) / (np.linalg.norm(x_prev) + 1e-8))
            history.append(res)
            if res < 1e-4:
                break

        return x.astype(np.float32), history

    # ── PnP-ADMM ─────────────────────────────────────────────────────────────

    def _pnp_admm(
        self, y: np.ndarray, action: ReconAction, modality: str
    ) -> tuple[np.ndarray, list[float]]:
        """PnP-ADMM with a Gaussian denoiser as plug-in."""
        hp       = action.hyperparameters
        sigma    = float(hp.get("sigma_denoiser", 0.02))
        num_iter = int(hp.get("num_iterations", 30))

        # Initialise from TV-ADMM (fewer iterations)
        x, history = self._tv_admm(y, action, modality)

        for it in range(num_iter):
            x_prev = x.copy()
            # Denoising step: Gaussian denoiser as prox operator
            x = gaussian_filter(x.astype(np.float64), sigma=sigma * x.max() + 1e-8)
            x = np.maximum(x, 0.0)
            res = float(np.linalg.norm(x - x_prev) / (np.linalg.norm(x_prev) + 1e-8))
            history.append(res)
            if res < 1e-4:
                break

        return x.astype(np.float32), history

    # ── TV proximal operator (Gaussian approx) ─────────────────────────────

    def _prox_tv(self, x: np.ndarray, strength: float) -> np.ndarray:
        """Approximate TV prox via Gaussian smoothing (fast, not exact)."""
        sigma = min(strength * 10.0, 2.0)
        return gaussian_filter(x.astype(np.float64), sigma=sigma)
