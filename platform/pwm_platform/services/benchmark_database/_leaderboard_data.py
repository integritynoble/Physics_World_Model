"""Leaderboard data keyed by variant — existing 4 variants have InverseNet data; new variants start empty."""

from __future__ import annotations

LEADERBOARD_DATA: dict[str, dict[str, list[dict]]] = {
    "sd_cassi": {
        "b2": [
            {"rank": 1, "method": "MST-L", "psnr": 34.81, "ssim": 0.973, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HDNet", "psnr": 34.66, "ssim": 0.970, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-HSICNN", "psnr": 25.12, "ssim": 0.832, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "GAP-TV", "psnr": 24.34, "ssim": 0.815, "source": "InverseNet", "adopted": False},
        ],
        "b4": [
            {"rank": 1, "method": "MST-L", "psnr": 27.33, "ssim": 0.881, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HDNet", "psnr": 27.10, "ssim": 0.875, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-HSICNN", "psnr": 22.45, "ssim": 0.780, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "GAP-TV", "psnr": 20.83, "ssim": 0.744, "source": "InverseNet", "adopted": False},
        ],
    },
    "cacti": {
        "b2": [
            {"rank": 1, "method": "EfficientSCI", "psnr": 35.39, "ssim": 0.973, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "ELP-Unfolding", "psnr": 34.09, "ssim": 0.965, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-FFDNet", "psnr": 29.28, "ssim": 0.910, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "GAP-TV", "psnr": 26.75, "ssim": 0.870, "source": "InverseNet", "adopted": False},
        ],
        "b4": [
            {"rank": 1, "method": "EfficientSCI", "psnr": 27.38, "ssim": 0.927, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "ELP-Unfolding", "psnr": 26.50, "ssim": 0.910, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-FFDNet", "psnr": 20.15, "ssim": 0.650, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "GAP-TV", "psnr": 14.81, "ssim": 0.303, "source": "InverseNet", "adopted": False},
        ],
    },
    "spc_block": {
        "b2": [
            {"rank": 1, "method": "ISTA-Net", "psnr": 31.85, "ssim": 0.916, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HATNet", "psnr": 30.98, "ssim": 0.905, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-DRUNet", "psnr": 30.53, "ssim": 0.895, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "FISTA-TV", "psnr": 28.06, "ssim": 0.850, "source": "InverseNet", "adopted": False},
        ],
        "b4": [
            {"rank": 1, "method": "ISTA-Net", "psnr": 27.45, "ssim": 0.760, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HATNet", "psnr": 26.80, "ssim": 0.745, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-DRUNet", "psnr": 24.10, "ssim": 0.690, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "FISTA-TV", "psnr": 19.02, "ssim": 0.584, "source": "InverseNet", "adopted": False},
        ],
    },
    "spc_kronecker": {
        "b2": [
            {"rank": 1, "method": "ISTA-Net", "psnr": 31.85, "ssim": 0.916, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HATNet", "psnr": 30.98, "ssim": 0.905, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-DRUNet", "psnr": 30.53, "ssim": 0.895, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "FISTA-TV", "psnr": 28.06, "ssim": 0.850, "source": "InverseNet", "adopted": False},
        ],
        "b4": [
            {"rank": 1, "method": "ISTA-Net", "psnr": 27.45, "ssim": 0.760, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HATNet", "psnr": 26.80, "ssim": 0.745, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-DRUNet", "psnr": 24.10, "ssim": 0.690, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "FISTA-TV", "psnr": 19.02, "ssim": 0.584, "source": "InverseNet", "adopted": False},
        ],
    },
}
