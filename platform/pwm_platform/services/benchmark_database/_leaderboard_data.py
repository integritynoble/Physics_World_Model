"""Leaderboard data keyed by variant — InverseNet validated baselines for CASSI, CACTI, SPC."""

from __future__ import annotations

# fmt: off

LEADERBOARD_DATA: dict[str, dict[str, list[dict]]] = {

    # ══════════════════════════════════════════════════════════════════════════
    #  SD-CASSI — Evaluated results (GAP-TV, MST-L, SSR-L on 3-tier challenge)
    #  Public: 10 KAIST scenes 256×256, clean step=2 (v2.0 regenerated)
    #  Dev: 20 PnP-CASSI crops 500×500, mismatch slope=2.08
    #  Hidden: 20 Pavia crops 500×500, mismatch slope=1.95
    #  Oracle mask correction applied for dev/hidden tiers
    # ══════════════════════════════════════════════════════════════════════════

    "sd_cassi": {
        # Challenge leaderboard — Blind Reconstruction Challenge (3-tier)
        # Scores: 0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × consistency; overall = mean(public, dev, hidden)
        "challenge": [
            {"rank": 1, "method": "SSR-L + gradient",      "public_score": 0.877, "dev_score": 0.456, "hidden_score": 0.545, "overall_score": 0.626, "details": {"public": {"psnr": 38.03, "ssim": 0.994, "consistency": 1.00}, "dev": {"psnr": 17.06, "ssim": 0.473, "consistency": 0.98}, "hidden": {"psnr": 19.53, "ssim": 0.626, "consistency": 1.00}}, "source": "PWM benchmark (CVPR 2024)"},
            {"rank": 2, "method": "GAP-TV + gradient",     "public_score": 0.687, "dev_score": 0.516, "hidden_score": 0.576, "overall_score": 0.593, "details": {"public": {"psnr": 24.21, "ssim": 0.865, "consistency": 1.00}, "dev": {"psnr": 18.37, "ssim": 0.583, "consistency": 1.00}, "hidden": {"psnr": 19.69, "ssim": 0.699, "consistency": 1.00}}, "source": "PWM benchmark"},
            {"rank": 3, "method": "MST-L + gradient",      "public_score": 0.794, "dev_score": 0.385, "hidden_score": 0.472, "overall_score": 0.550, "details": {"public": {"psnr": 31.29, "ssim": 0.977, "consistency": 0.95}, "dev": {"psnr": 15.45, "ssim": 0.384, "consistency": 0.88}, "hidden": {"psnr": 17.18, "ssim": 0.550, "consistency": 0.90}}, "source": "PWM benchmark"},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  CACTI — Competition results (6 CACTI scenes, 20 samples, 5 algorithms)
    #  Dataset v3.0: paper's exact affine_transform warp, peak_photon=10000
    #  Public: Scenario III (Oracle) results — paper's exact forward model
    #  Dev/Hidden: pending procedural tier evaluation
    #  Benchmark run 2026-03-01, ELP-Unfolding on Modal A10G, rest local
    # ══════════════════════════════════════════════════════════════════════════

    "cacti": {
        "challenge": [
            {"rank": 1, "method": "EfficientSCI + oracle",            "public_score": 0.740, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.247, "details": {"public": {"psnr": 27.71, "ssim": 0.930, "consistency": 0.993}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario III (affine_transform warp, public tier)"},
            {"rank": 2, "method": "HiSViT-9 + oracle",                "public_score": 0.740, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.247, "details": {"public": {"psnr": 27.62, "ssim": 0.934, "consistency": 0.993}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario III (affine_transform warp, public tier)"},
            {"rank": 3, "method": "ELP-Unfolding + oracle",            "public_score": 0.725, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.242, "details": {"public": {"psnr": 29.34, "ssim": 0.926, "consistency": 0.816}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario III (Modal A10G, affine_transform warp)"},
            {"rank": 4, "method": "GAP-TV + oracle",                   "public_score": 0.631, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.210, "details": {"public": {"psnr": 24.40, "ssim": 0.764, "consistency": 1.000}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario III (affine_transform warp, public tier)"},
            {"rank": 5, "method": "PnP-DnCNN + oracle",               "public_score": 0.617, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.206, "details": {"public": {"psnr": 23.41, "ssim": 0.772, "consistency": 0.981}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario III (affine_transform warp, public tier)"},
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
