"""Leaderboard data keyed by variant — InverseNet validated baselines for CASSI, CACTI, SPC."""

from __future__ import annotations

# fmt: off

LEADERBOARD_DATA: dict[str, dict[str, list[dict]]] = {

    # ══════════════════════════════════════════════════════════════════════════
    #  SD-CASSI — Evaluated results (5 solvers on 3-tier challenge)
    #  Public: 10 KAIST scenes 256×256, clean step=2 (v2.0 regenerated)
    #  Dev: 20 Pavia crops 500×500, mismatch slope=1.95
    #  Hidden: 20 PnP-CASSI crops 500×500, mismatch slope=2.08
    #  Oracle mask correction applied for dev/hidden tiers
    # ══════════════════════════════════════════════════════════════════════════

    "sd_cassi": {
        # Normal leaderboard — Standard reconstruction (known forward model, no mismatch)
        # Dataset: 10 KAIST simulation scenes, 256×256×28 spectral channels
        # Score: 0.5 × clip((PSNR−15)/30, 0, 1) + 0.5 × SSIM  (no consistency term)
        # Sources: MST (CVPR 2022), CST (CVPR 2022), HDNet (ECCV 2022), others
        "normal": [
            {"rank": 1, "method": "HDNet",       "psnr": 35.14, "ssim": 0.952, "score": 0.812, "dataset": "KAIST simu, 256×256×28", "source": "HDNet (ECCV 2022)"},
            {"rank": 2, "method": "MST-L",       "psnr": 33.37, "ssim": 0.949, "score": 0.781, "dataset": "KAIST simu, 256×256×28", "source": "MST (CVPR 2022)"},
            {"rank": 3, "method": "CST-L",       "psnr": 32.74, "ssim": 0.948, "score": 0.770, "dataset": "KAIST simu, 256×256×28", "source": "CST (CVPR 2022)"},
            {"rank": 4, "method": "TSA-Net",     "psnr": 31.46, "ssim": 0.894, "score": 0.721, "dataset": "KAIST simu, 256×256×28", "source": "TSA-Net (ECCV 2020)"},
            {"rank": 5, "method": "PnP-HSI-DIP", "psnr": 30.31, "ssim": 0.851, "score": 0.681, "dataset": "KAIST simu, 256×256×28", "source": "PnP-HSI (ICCV 2021)"},
            {"rank": 6, "method": "GAP-TV",      "psnr": 26.86, "ssim": 0.861, "score": 0.629, "dataset": "KAIST simu, 256×256×28", "source": "GAP-TV (Signal Processing 2016)"},
            {"rank": 7, "method": "DeSCI",       "psnr": 27.13, "ssim": 0.841, "score": 0.623, "dataset": "KAIST simu, 256×256×28", "source": "DeSCI (TPAMI 2019)"},
        ],
        # Challenge leaderboard — Blind Reconstruction Challenge (3-tier)
        # Scores: 0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × consistency; overall = mean(public, dev, hidden)
        "challenge": [
            {"rank": 1, "method": "SSR-L + gradient",      "public_score": 0.877, "dev_score": 0.545, "hidden_score": 0.456, "overall_score": 0.626, "details": {"public": {"psnr": 38.03, "ssim": 0.994, "consistency": 1.00}, "dev": {"psnr": 19.53, "ssim": 0.626, "consistency": 1.00}, "hidden": {"psnr": 17.06, "ssim": 0.473, "consistency": 0.98}}, "source": "PWM benchmark (CVPR 2024)"},
            {"rank": 2, "method": "GAP-TV + gradient",     "public_score": 0.687, "dev_score": 0.576, "hidden_score": 0.516, "overall_score": 0.593, "details": {"public": {"psnr": 24.21, "ssim": 0.865, "consistency": 1.00}, "dev": {"psnr": 19.69, "ssim": 0.699, "consistency": 1.00}, "hidden": {"psnr": 18.37, "ssim": 0.583, "consistency": 1.00}}, "source": "PWM benchmark"},
            {"rank": 3, "method": "MST-L + gradient",      "public_score": 0.794, "dev_score": 0.472, "hidden_score": 0.385, "overall_score": 0.550, "details": {"public": {"psnr": 31.29, "ssim": 0.977, "consistency": 0.95}, "dev": {"psnr": 17.18, "ssim": 0.550, "consistency": 0.90}, "hidden": {"psnr": 15.45, "ssim": 0.384, "consistency": 0.88}}, "source": "PWM benchmark"},
            {"rank": 4, "method": "PnP-HSICNN + gradient", "public_score": 0.549, "dev_score": 0.508, "hidden_score": 0.455, "overall_score": 0.504, "details": {"public": {"psnr": 20.06, "ssim": 0.675, "consistency": 0.89}, "dev": {"psnr": 16.88, "ssim": 0.621, "consistency": 0.95}, "hidden": {"psnr": 15.59, "ssim": 0.518, "consistency": 0.96}}, "source": "PWM benchmark"},
            {"rank": 5, "method": "HDNet + gradient",      "public_score": 0.707, "dev_score": 0.329, "hidden_score": 0.280, "overall_score": 0.439, "details": {"public": {"psnr": 25.45, "ssim": 0.921, "consistency": 0.92}, "dev": {"psnr": 13.59, "ssim": 0.427, "consistency": 0.61}, "hidden": {"psnr": 10.96, "ssim": 0.373, "consistency": 0.61}}, "source": "PWM benchmark"},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  CACTI — Competition results (6 CACTI scenes, 20 samples, 5 algorithms)
    #  Dataset v3.0: paper's exact affine_transform warp, peak_photon=10000
    #  Public: Scenario IV (Blind Cal) — hybrid accumulated grid search
    #  Dev/Hidden: pending (512×512×8, ~2h calibration per tier)
    #  Benchmark run 2026-03-02, improved blind calibration (hybrid mask)
    # ══════════════════════════════════════════════════════════════════════════

    "cacti": {
        # Normal leaderboard — Standard video SCI reconstruction (known mask, no mismatch)
        # Dataset: 6 CACTI simulation scenes (kobe, runner, drop, crash, aerial, vehicle)
        # T=8 frames, Cr=8 compression ratio; score = 0.5 × clip((PSNR−15)/30, 0, 1) + 0.5 × SSIM
        # Sources: EfficientSCI (CVPR 2023), HiSViT (ECCV 2024), RevSCI (TPAMI 2022), ELP (2022)
        "normal": [
            {"rank": 1, "method": "HiSViT-9",       "psnr": 38.24, "ssim": 0.978, "score": 0.876, "dataset": "6-scene simulation, T=8, Cr=8", "source": "HiSViT (ECCV 2024)"},
            {"rank": 2, "method": "EfficientSCI",   "psnr": 37.71, "ssim": 0.976, "score": 0.867, "dataset": "6-scene simulation, T=8, Cr=8", "source": "EfficientSCI (CVPR 2023)"},
            {"rank": 3, "method": "ELP-Unfolding",  "psnr": 35.54, "ssim": 0.968, "score": 0.826, "dataset": "6-scene simulation, T=8, Cr=8", "source": "ELP-Unfolding (2022)"},
            {"rank": 4, "method": "RevSCI",         "psnr": 33.49, "ssim": 0.956, "score": 0.786, "dataset": "6-scene simulation, T=8, Cr=8", "source": "RevSCI (TPAMI 2022)"},
            {"rank": 5, "method": "BIRNAT",         "psnr": 30.26, "ssim": 0.921, "score": 0.715, "dataset": "6-scene simulation, T=8, Cr=8", "source": "BIRNAT (TPAMI 2021)"},
            {"rank": 6, "method": "GAP-TV",         "psnr": 26.02, "ssim": 0.892, "score": 0.630, "dataset": "6-scene simulation, T=8, Cr=8", "source": "GAP-TV (Signal Processing 2016)"},
        ],
        "challenge": [
            {"rank": 1, "method": "ELP-Unfolding + blind cal",        "public_score": 0.572, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.191, "details": {"public": {"psnr": 23.06, "ssim": 0.672, "consistency": 0.977}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario IV (blind cal, improved hybrid grid search)"},
            {"rank": 2, "method": "EfficientSCI + blind cal",         "public_score": 0.567, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.189, "details": {"public": {"psnr": 22.65, "ssim": 0.676, "consistency": 0.974}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario IV (blind cal, improved hybrid grid search)"},
            {"rank": 3, "method": "HiSViT-9 + blind cal",             "public_score": 0.565, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.188, "details": {"public": {"psnr": 22.58, "ssim": 0.673, "consistency": 0.974}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario IV (blind cal, improved hybrid grid search)"},
            {"rank": 4, "method": "GAP-TV + blind cal",               "public_score": 0.522, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.174, "details": {"public": {"psnr": 21.79, "ssim": 0.590, "consistency": 0.977}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario IV (blind cal, improved hybrid grid search)"},
            {"rank": 5, "method": "PnP-FFDNet + blind cal",           "public_score": 0.446, "dev_score": 0.000, "hidden_score": 0.000, "overall_score": 0.149, "details": {"public": {"psnr": 17.37, "ssim": 0.542, "consistency": 0.987}, "dev": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}, "hidden": {"psnr": 0.0, "ssim": 0.0, "consistency": 0.0}}, "source": "InverseNet Scenario IV (blind cal, improved hybrid grid search)"},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  SPC — Block — InverseNet validated results (11 Set11 images, 4 algorithms)
    # ══════════════════════════════════════════════════════════════════════════

    "spc_block": {
        # Normal leaderboard — Standard CS reconstruction (known sensing matrix, no mismatch)
        # Dataset: Set11 (11 grayscale images), 33×33 blocks, Gaussian sensing matrix, CR=0.25
        # Score: 0.5 × clip((PSNR−15)/30, 0, 1) + 0.5 × SSIM
        "normal": [
            {"rank": 1, "method": "HATNet",     "psnr": 28.85, "ssim": 0.801, "score": 0.731, "dataset": "Set11, block 33×33, CR=0.25", "source": "InverseNet baseline (oracle cal)"},
            {"rank": 2, "method": "ISTA-Net+",  "psnr": 27.56, "ssim": 0.762, "score": 0.710, "dataset": "Set11, block 33×33, CR=0.25", "source": "ISTA-Net+ (CVPR 2018)"},
            {"rank": 3, "method": "FISTA-TV",   "psnr": 26.10, "ssim": 0.758, "score": 0.686, "dataset": "Set11, block 33×33, CR=0.25", "source": "InverseNet baseline (oracle cal)"},
            {"rank": 4, "method": "PnP-DRUNet", "psnr": 23.85, "ssim": 0.665, "score": 0.632, "dataset": "Set11, block 33×33, CR=0.25", "source": "InverseNet baseline (oracle cal)"},
        ],
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
        # Normal leaderboard — Standard CS reconstruction (known Kronecker sensing matrix, no mismatch)
        # Dataset: Set11 (11 grayscale images), Kronecker sensing, CR=0.25
        # Score: 0.5 × clip((PSNR−15)/30, 0, 1) + 0.5 × SSIM
        "normal": [
            {"rank": 1, "method": "PnP-DRUNet",        "psnr": 27.85, "ssim": 0.837, "score": 0.753, "dataset": "Set11, Kronecker, CR=0.25", "source": "InverseNet baseline (oracle cal)"},
            {"rank": 2, "method": "FISTA-TV (tuned)",  "psnr": 26.71, "ssim": 0.804, "score": 0.727, "dataset": "Set11, Kronecker, CR=0.25", "source": "InverseNet baseline (oracle cal)"},
            {"rank": 3, "method": "HATNet + FISTA-TV", "psnr": 26.45, "ssim": 0.790, "score": 0.720, "dataset": "Set11, Kronecker, CR=0.25", "source": "InverseNet baseline (oracle cal)"},
            {"rank": 4, "method": "FISTA-TV (paper)",  "psnr": 25.98, "ssim": 0.776, "score": 0.708, "dataset": "Set11, Kronecker, CR=0.25", "source": "InverseNet baseline (oracle cal)"},
            {"rank": 5, "method": "ISTA-Net",          "psnr": 26.40, "ssim": 0.721, "score": 0.710, "dataset": "Set11, Kronecker, CR=0.25", "source": "InverseNet baseline (oracle cal)"},
            {"rank": 6, "method": "PnP-BM3D",          "psnr": 21.10, "ssim": 0.582, "score": 0.601, "dataset": "Set11, Kronecker, CR=0.25", "source": "InverseNet baseline (oracle cal)"},
        ],
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
