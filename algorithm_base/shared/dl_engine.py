"""Deep learning reconstruction engine using deepinv pretrained models.

Provides real pretrained DRUNet/DnCNN-based PnP reconstruction with
configurable optimizers (PGD, HQS, DRS/ADMM) and hyperparameters so that
each DL algorithm produces genuinely different PSNR/SSIM values.
"""

import numpy as np
import torch
from scipy.signal import fftconvolve

_DRUNET = None
_DNCNN = None
_DEVICE = None


def _get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def _get_drunet():
    global _DRUNET
    if _DRUNET is None:
        import deepinv.models
        dev = _get_device()
        _DRUNET = deepinv.models.DRUNet(
            pretrained="download", in_channels=1, out_channels=1, device=dev
        )
        _DRUNET.eval()
    return _DRUNET


def _get_dncnn():
    global _DNCNN
    if _DNCNN is None:
        import deepinv.models
        dev = _get_device()
        _DNCNN = deepinv.models.DnCNN(
            pretrained="download", in_channels=1, out_channels=1, device=dev
        )
        _DNCNN.eval()
    return _DNCNN


def _make_psf_physics(psf_sigma, img_size=256):
    import deepinv.physics
    k = max(3, int(3 * psf_sigma))
    ax = np.arange(-k, k + 1)
    gx, gy = np.meshgrid(ax, ax)
    psf = np.exp(-(gx ** 2 + gy ** 2) / (2 * psf_sigma ** 2)).astype(np.float32)
    psf /= psf.sum()
    psf_t = torch.tensor(psf).unsqueeze(0).unsqueeze(0)
    dev = _get_device()
    return deepinv.physics.BlurFFT(
        img_size=(1, img_size, img_size), filter=psf_t, device=dev
    )


def dl_pnp_drunet(y, psf_sigma=3.0, optimizer="PGD", sigma=0.05,
                  max_iter=10, stepsize=1.0):
    """PnP reconstruction with pretrained DRUNet denoiser."""
    import deepinv.optim
    dev = _get_device()
    denoiser = _get_drunet()
    physics = _make_psf_physics(psf_sigma)
    y_t = torch.tensor(y.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)
    model = deepinv.optim.optim_builder(
        iteration=optimizer,
        prior=deepinv.optim.PnP(denoiser=denoiser),
        data_fidelity=deepinv.optim.L2(),
        max_iter=max_iter,
        params_algo={"stepsize": stepsize, "g_param": sigma},
    )
    with torch.no_grad():
        x_hat = model(y_t, physics)
    return np.clip(x_hat.cpu().squeeze().numpy(), 0, 1).astype(np.float32)


def dl_red_drunet(y, psf_sigma=3.0, sigma=0.05, max_iter=10,
                  stepsize=0.5, lam=1.0):
    """RED (Regularization by Denoising) with pretrained DRUNet."""
    import deepinv.optim
    dev = _get_device()
    denoiser = _get_drunet()
    physics = _make_psf_physics(psf_sigma)
    y_t = torch.tensor(y.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)
    model = deepinv.optim.optim_builder(
        iteration="PGD",
        prior=deepinv.optim.RED(denoiser=denoiser),
        data_fidelity=deepinv.optim.L2(),
        max_iter=max_iter,
        params_algo={"stepsize": stepsize, "g_param": sigma, "lambda": lam},
    )
    with torch.no_grad():
        x_hat = model(y_t, physics)
    return np.clip(x_hat.cpu().squeeze().numpy(), 0, 1).astype(np.float32)


def dl_dncnn_denoise(y, psf_sigma=3.0):
    """Direct DnCNN denoising on adjoint-initialized image."""
    dev = _get_device()
    model = _get_dncnn()
    k = max(3, int(3 * psf_sigma))
    ax = np.arange(-k, k + 1)
    gx, gy = np.meshgrid(ax, ax)
    psf = np.exp(-(gx ** 2 + gy ** 2) / (2 * psf_sigma ** 2)).astype(np.float32)
    psf /= psf.sum()
    x_adj = fftconvolve(y, psf[::-1, ::-1], mode="same")
    y_t = torch.tensor(x_adj.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = model(y_t)
    return np.clip(out.cpu().squeeze().numpy(), 0, 1).astype(np.float32)


def dl_drunet_denoise(y, psf_sigma=3.0, sigma=0.05):
    """Direct DRUNet denoising on input measurement."""
    dev = _get_device()
    model = _get_drunet()
    y_t = torch.tensor(y.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = model(y_t, sigma=torch.tensor([sigma]).to(dev))
    return np.clip(out.cpu().squeeze().numpy(), 0, 1).astype(np.float32)
