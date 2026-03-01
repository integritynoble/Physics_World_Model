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
    #  CACTI — Competition results (6 CACTI scenes + procedural, 5 algorithms)
    #  Redesigned mismatch params: public=mild, dev=moderate, hidden=hard
    #  Actual benchmark results from Modal GPU evaluation (2026-03-01)
    # ══════════════════════════════════════════════════════════════════════════

    "cacti": {
        "challenge": [
            {"rank": 1, "method": "EfficientSCI + blind cal",         "public_score": 0.623, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.208, "details": {"public": {"psnr": 24.09, "ssim": 0.778, "consistency": 0.95}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario IV (public only)"},
            {"rank": 2, "method": "ELP-Unfolding + blind cal",        "public_score": 0.573, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.191, "details": {"public": {"psnr": 21.92, "ssim": 0.703, "consistency": 1.00}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario IV (public only)"},
            {"rank": 3, "method": "PnP-DnCNN + blind cal",            "public_score": 0.573, "dev_score": 0.380, "hidden_score": 0.359, "overall_score": 0.437, "details": {"public": {"psnr": 21.54, "ssim": 0.724, "consistency": 0.98}, "dev": {"psnr": 17.23, "ssim": 0.385, "consistency": 0.98}, "hidden": {"psnr": 16.23, "ssim": 0.364, "consistency": 0.97}}, "source": "InverseNet Scenario IV"},
            {"rank": 4, "method": "GAP-TV + blind cal",               "public_score": 0.571, "dev_score": 0.393, "hidden_score": 0.367, "overall_score": 0.444, "details": {"public": {"psnr": 21.70, "ssim": 0.706, "consistency": 1.00}, "dev": {"psnr": 18.75, "ssim": 0.361, "consistency": 0.99}, "hidden": {"psnr": 17.71, "ssim": 0.337, "consistency": 0.98}}, "source": "InverseNet Scenario IV"},
            {"rank": 5, "method": "HiSViT-9 + blind cal",             "public_score": 0.000, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.000, "details": {"public": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario IV (pending GPU eval)"},
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
    #  SPC — Kronecker (11 Set11 images, 6 algorithms, Scenario IV blind cal)
    #  Actual benchmark results from Modal GPU evaluation (2026-03-01)
    # ══════════════════════════════════════════════════════════════════════════

    "spc_kronecker": {
        "challenge": [
            {"rank": 1, "method": "PnP-DRUNet + blind cal",         "public_score": 0.720, "dev_score": 0.736, "hidden_score": 0.687, "overall_score": 0.714, "details": {"public": {"psnr": 26.33, "ssim": 0.814, "consistency": 0.92}, "dev": {"psnr": 27.02, "ssim": 0.828, "consistency": 0.94}, "hidden": {"psnr": 24.75, "ssim": 0.776, "consistency": 0.90}}, "source": "InverseNet Scenario IV"},
            {"rank": 2, "method": "FISTA-TV (tuned) + blind cal",   "public_score": 0.693, "dev_score": 0.710, "hidden_score": 0.671, "overall_score": 0.691, "details": {"public": {"psnr": 25.34, "ssim": 0.757, "consistency": 0.94}, "dev": {"psnr": 25.93, "ssim": 0.781, "consistency": 0.95}, "hidden": {"psnr": 24.50, "ssim": 0.730, "consistency": 0.91}}, "source": "InverseNet Scenario IV"},
            {"rank": 3, "method": "FISTA-TV (paper) + blind cal",   "public_score": 0.690, "dev_score": 0.704, "hidden_score": 0.665, "overall_score": 0.686, "details": {"public": {"psnr": 25.21, "ssim": 0.751, "consistency": 0.94}, "dev": {"psnr": 25.75, "ssim": 0.767, "consistency": 0.96}, "hidden": {"psnr": 24.24, "ssim": 0.722, "consistency": 0.91}}, "source": "InverseNet Scenario IV"},
            {"rank": 4, "method": "HATNet + FISTA-TV + blind cal",  "public_score": 0.686, "dev_score": 0.702, "hidden_score": 0.665, "overall_score": 0.684, "details": {"public": {"psnr": 25.38, "ssim": 0.746, "consistency": 0.92}, "dev": {"psnr": 25.95, "ssim": 0.768, "consistency": 0.94}, "hidden": {"psnr": 24.53, "ssim": 0.720, "consistency": 0.90}}, "source": "InverseNet Scenario IV"},
            {"rank": 5, "method": "ISTA-Net + blind cal",           "public_score": 0.628, "dev_score": 0.686, "hidden_score": 0.509, "overall_score": 0.608, "details": {"public": {"psnr": 24.11, "ssim": 0.595, "consistency": 0.99}, "dev": {"psnr": 26.05, "ssim": 0.701, "consistency": 0.99}, "hidden": {"psnr": 19.99, "ssim": 0.385, "consistency": 0.98}}, "source": "InverseNet Scenario IV"},
            {"rank": 6, "method": "PnP-BM3D + blind cal",           "public_score": 0.565, "dev_score": 0.550, "hidden_score": 0.580, "overall_score": 0.565, "details": {"public": {"psnr": 19.57, "ssim": 0.527, "consistency": 0.99}, "dev": {"psnr": 18.36, "ssim": 0.511, "consistency": 0.99}, "hidden": {"psnr": 20.53, "ssim": 0.561, "consistency": 0.96}}, "source": "InverseNet Scenario IV"},
        ],
    },
}
