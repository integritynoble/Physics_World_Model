"""Leaderboard data keyed by variant — InverseNet validated baselines for CASSI, CACTI, SPC."""

from __future__ import annotations

# fmt: off

LEADERBOARD_DATA: dict[str, dict[str, list[dict]]] = {

    # ══════════════════════════════════════════════════════════════════════════
    #  SD-CASSI — InverseNet validated results (10 KAIST scenes, 4 algorithms)
    # ══════════════════════════════════════════════════════════════════════════

    "sd_cassi": {
        # Challenge leaderboard — Blind Reconstruction Challenge (3-tier)
        # Scores: 0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × consistency; overall = mean(public, dev, hidden)
        "challenge": [
            {"rank": 1, "method": "MST-L + gradient",     "public_score": 0.751, "dev_score": 0.706, "hidden_score": 0.689, "overall_score": 0.715, "details": {"public": {"psnr": 26.88, "ssim": 0.871, "consistency": 0.94}, "dev": {"psnr": 24.73, "ssim": 0.826, "consistency": 0.89}, "hidden": {"psnr": 24.08, "ssim": 0.812, "consistency": 0.86}}, "source": "InverseNet baseline"},
            {"rank": 2, "method": "HDNet + gradient",      "public_score": 0.653, "dev_score": 0.637, "hidden_score": 0.631, "overall_score": 0.640, "details": {"public": {"psnr": 21.88, "ssim": 0.756, "consistency": 0.88}, "dev": {"psnr": 21.88, "ssim": 0.756, "consistency": 0.80}, "hidden": {"psnr": 21.88, "ssim": 0.756, "consistency": 0.77}}, "source": "InverseNet baseline"},
            {"rank": 3, "method": "PnP-HSICNN + gradient", "public_score": 0.645, "dev_score": 0.608, "hidden_score": 0.594, "overall_score": 0.616, "details": {"public": {"psnr": 22.84, "ssim": 0.690, "consistency": 0.93}, "dev": {"psnr": 21.90, "ssim": 0.646, "consistency": 0.87}, "hidden": {"psnr": 21.63, "ssim": 0.633, "consistency": 0.84}}, "source": "InverseNet baseline"},
            {"rank": 4, "method": "GAP-TV + gradient",     "public_score": 0.627, "dev_score": 0.600, "hidden_score": 0.590, "overall_score": 0.606, "details": {"public": {"psnr": 21.63, "ssim": 0.679, "consistency": 0.91}, "dev": {"psnr": 21.36, "ssim": 0.652, "consistency": 0.84}, "hidden": {"psnr": 21.28, "ssim": 0.644, "consistency": 0.81}}, "source": "InverseNet baseline"},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  CACTI — InverseNet validated results (6 DAVIS videos, 4 algorithms)
    # ══════════════════════════════════════════════════════════════════════════

    "cacti": {
        "challenge": [
            {"rank": 1, "method": "ELP-Unfolding + gradient", "public_score": 0.754, "dev_score": 0.616, "hidden_score": 0.575, "overall_score": 0.648, "details": {"public": {"psnr": 28.01, "ssim": 0.865, "consistency": 0.92}, "dev": {"psnr": 23.13, "ssim": 0.648, "consistency": 0.86}, "hidden": {"psnr": 21.74, "ssim": 0.587, "consistency": 0.83}}, "source": "InverseNet baseline"},
            {"rank": 2, "method": "EfficientSCI + gradient",  "public_score": 0.739, "dev_score": 0.604, "hidden_score": 0.563, "overall_score": 0.635, "details": {"public": {"psnr": 26.12, "ssim": 0.865, "consistency": 0.92}, "dev": {"psnr": 21.72, "ssim": 0.646, "consistency": 0.86}, "hidden": {"psnr": 20.47, "ssim": 0.584, "consistency": 0.83}}, "source": "InverseNet baseline"},
            {"rank": 3, "method": "GAP-TV + gradient",        "public_score": 0.674, "dev_score": 0.560, "hidden_score": 0.527, "overall_score": 0.587, "details": {"public": {"psnr": 24.79, "ssim": 0.735, "consistency": 0.91}, "dev": {"psnr": 21.11, "ssim": 0.559, "consistency": 0.84}, "hidden": {"psnr": 20.09, "ssim": 0.510, "consistency": 0.81}}, "source": "InverseNet baseline"},
            {"rank": 4, "method": "PnP-FFDNet + gradient",    "public_score": 0.685, "dev_score": 0.550, "hidden_score": 0.508, "overall_score": 0.581, "details": {"public": {"psnr": 24.13, "ssim": 0.766, "consistency": 0.93}, "dev": {"psnr": 19.25, "ssim": 0.554, "consistency": 0.87}, "hidden": {"psnr": 17.85, "ssim": 0.494, "consistency": 0.84}}, "source": "InverseNet baseline"},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  SPC — Block — InverseNet validated results (11 Set11 images, 4 algorithms)
    # ══════════════════════════════════════════════════════════════════════════

    "spc_block": {
        "challenge": [
            {"rank": 1, "method": "HATNet + gradient",     "public_score": 0.710, "dev_score": 0.630, "hidden_score": 0.612, "overall_score": 0.651, "details": {"public": {"psnr": 27.91, "ssim": 0.778, "consistency": 0.88}, "dev": {"psnr": 23.34, "ssim": 0.708, "consistency": 0.80}, "hidden": {"psnr": 22.51, "ssim": 0.696, "consistency": 0.77}}, "source": "InverseNet baseline"},
            {"rank": 2, "method": "ISTA-Net + gradient",   "public_score": 0.694, "dev_score": 0.634, "hidden_score": 0.614, "overall_score": 0.647, "details": {"public": {"psnr": 26.61, "ssim": 0.742, "consistency": 0.92}, "dev": {"psnr": 23.66, "ssim": 0.681, "consistency": 0.86}, "hidden": {"psnr": 22.81, "ssim": 0.663, "consistency": 0.83}}, "source": "InverseNet baseline"},
            {"rank": 3, "method": "FISTA-TV + gradient",   "public_score": 0.680, "dev_score": 0.618, "hidden_score": 0.600, "overall_score": 0.633, "details": {"public": {"psnr": 25.29, "ssim": 0.738, "consistency": 0.91}, "dev": {"psnr": 22.51, "ssim": 0.676, "consistency": 0.84}, "hidden": {"psnr": 21.74, "ssim": 0.659, "consistency": 0.81}}, "source": "InverseNet baseline"},
            {"rank": 4, "method": "PnP-DRUNet + gradient", "public_score": 0.627, "dev_score": 0.560, "hidden_score": 0.537, "overall_score": 0.575, "details": {"public": {"psnr": 22.99, "ssim": 0.643, "consistency": 0.93}, "dev": {"psnr": 20.41, "ssim": 0.556, "consistency": 0.87}, "hidden": {"psnr": 19.68, "ssim": 0.530, "consistency": 0.84}}, "source": "InverseNet baseline"},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  SPC — Kronecker (same Set11 baselines, different measurement matrix)
    # ══════════════════════════════════════════════════════════════════════════

    "spc_kronecker": {
        "challenge": [
            {"rank": 1, "method": "HATNet + gradient",     "public_score": 0.710, "dev_score": 0.630, "hidden_score": 0.612, "overall_score": 0.651, "details": {"public": {"psnr": 27.91, "ssim": 0.778, "consistency": 0.88}, "dev": {"psnr": 23.34, "ssim": 0.708, "consistency": 0.80}, "hidden": {"psnr": 22.51, "ssim": 0.696, "consistency": 0.77}}, "source": "InverseNet baseline"},
            {"rank": 2, "method": "ISTA-Net + gradient",   "public_score": 0.694, "dev_score": 0.634, "hidden_score": 0.614, "overall_score": 0.647, "details": {"public": {"psnr": 26.61, "ssim": 0.742, "consistency": 0.92}, "dev": {"psnr": 23.66, "ssim": 0.681, "consistency": 0.86}, "hidden": {"psnr": 22.81, "ssim": 0.663, "consistency": 0.83}}, "source": "InverseNet baseline"},
            {"rank": 3, "method": "FISTA-TV + gradient",   "public_score": 0.680, "dev_score": 0.618, "hidden_score": 0.600, "overall_score": 0.633, "details": {"public": {"psnr": 25.29, "ssim": 0.738, "consistency": 0.91}, "dev": {"psnr": 22.51, "ssim": 0.676, "consistency": 0.84}, "hidden": {"psnr": 21.74, "ssim": 0.659, "consistency": 0.81}}, "source": "InverseNet baseline"},
            {"rank": 4, "method": "PnP-DRUNet + gradient", "public_score": 0.627, "dev_score": 0.560, "hidden_score": 0.537, "overall_score": 0.575, "details": {"public": {"psnr": 22.99, "ssim": 0.643, "consistency": 0.93}, "dev": {"psnr": 20.41, "ssim": 0.556, "consistency": 0.87}, "hidden": {"psnr": 19.68, "ssim": 0.530, "consistency": 0.84}}, "source": "InverseNet baseline"},
        ],
    },
}
