"""Challenge configuration — Blind Reconstruction Challenge (Benchmark New).

Each variant entry defines scoring weights, spec ranges (what contestants see),
true spec (server-side only), dataset splits (Pro / Hidden), and baseline
performance from InverseNet Scenarios II and III.
"""

from __future__ import annotations

# fmt: off

CHALLENGE_CONFIG: dict[str, dict] = {

    # ══════════════════════════════════════════════════════════════════════════
    #  SD-CASSI — 10 KAIST hyperspectral scenes (256×256×28)
    # ══════════════════════════════════════════════════════════════════════════

    "sd_cassi": {
        "scoring": {
            "psnr_weight": 0.40,
            "ssim_weight": 0.40,
            "consistency_weight": 0.20,
            "formula_display": "0.4 \u00d7 PSNR_norm + 0.4 \u00d7 SSIM + 0.2 \u00d7 (1 \u2212 \u2016y \u2212 H\u0302x\u0302\u2016/\u2016y\u2016)",
        },
        "spec_ranges": [
            {"name": "mask_dx",          "min": 0.3,  "max": 0.7,  "unit": "px"},
            {"name": "mask_dy",          "min": 0.1,  "max": 0.5,  "unit": "px"},
            {"name": "mask_rotation",    "min": 0.0,  "max": 0.2,  "unit": "deg"},
            {"name": "dispersion_slope", "min": 1.90, "max": 2.15, "unit": "px/band"},
            {"name": "dispersion_axis",  "min": 0.0,  "max": 0.3,  "unit": "deg"},
        ],
        "true_spec": {
            "mask_dx": 0.5,
            "mask_dy": 0.3,
            "mask_rotation": 0.1,
            "dispersion_slope": 2.02,
            "dispersion_axis": 0.15,
        },
        "noise_model": "poisson_gaussian",
        "noise_params": {"poisson_alpha": 1.0, "gaussian_sigma": 0.01},
        "splits": {
            "pro":    {"scenes": [1, 2, 3, 4, 5, 6], "count": 6},
            "hidden": {"scenes": [7, 8, 9, 10],      "count": 4},
        },
        "data_source": "datasets/TSA_simu_data/Truth/",
        "data_format": "mat",
        "signal_shape": [256, 256, 28],
        "baselines": {
            "scenario_ii": [
                {"method": "MST-L",      "psnr": 20.83, "ssim": 0.744},
                {"method": "HDNet",       "psnr": 21.88, "ssim": 0.756},
                {"method": "PnP-HSICNN",  "psnr": 20.40, "ssim": 0.574},
                {"method": "GAP-TV",      "psnr": 20.96, "ssim": 0.612},
            ],
            "scenario_iii": [
                {"method": "MST-L",      "psnr": 27.33, "ssim": 0.881},
                {"method": "HDNet",       "psnr": 21.88, "ssim": 0.756},
                {"method": "PnP-HSICNN",  "psnr": 23.08, "ssim": 0.702},
                {"method": "GAP-TV",      "psnr": 21.72, "ssim": 0.688},
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  CACTI — 6 synthetic video scenes (256×256×8 frames)
    # ══════════════════════════════════════════════════════════════════════════

    "cacti": {
        "scoring": {
            "psnr_weight": 0.40,
            "ssim_weight": 0.40,
            "consistency_weight": 0.20,
            "formula_display": "0.4 \u00d7 PSNR_norm + 0.4 \u00d7 SSIM + 0.2 \u00d7 (1 \u2212 \u2016y \u2212 H\u0302x\u0302\u2016/\u2016y\u2016)",
        },
        "spec_ranges": [
            {"name": "mask_dx",       "min": 0.2,  "max": 0.8,  "unit": "px"},
            {"name": "mask_dy",       "min": 0.1,  "max": 0.5,  "unit": "px"},
            {"name": "mask_rotation", "min": 0.0,  "max": 0.3,  "unit": "deg"},
            {"name": "mask_blur",     "min": 0.0,  "max": 0.5,  "unit": "px"},
            {"name": "clock_offset",  "min": -0.1, "max": 0.1,  "unit": "frames"},
            {"name": "gain_drift",    "min": 0.95, "max": 1.05, "unit": ""},
            {"name": "offset_drift",  "min": -0.02, "max": 0.02, "unit": ""},
        ],
        "true_spec": {
            "mask_dx": 0.5,
            "mask_dy": 0.3,
            "mask_rotation": 0.15,
            "mask_blur": 0.2,
            "clock_offset": 0.05,
            "gain_drift": 1.02,
            "offset_drift": 0.01,
        },
        "noise_model": "poisson_gaussian",
        "noise_params": {"poisson_alpha": 1.0, "gaussian_sigma": 0.01},
        "splits": {
            "pro":    {"scenes": ["kobe", "traffic", "crash", "aerial"], "count": 4},
            "hidden": {"scenes": ["runner", "drop"],                     "count": 2},
        },
        "data_source": "datasets/CACTI/simulation/",
        "data_format": "mat",
        "signal_shape": [256, 256, 8],
        "baselines": {
            "scenario_ii": [
                {"method": "EfficientSCI",  "psnr": 27.38, "ssim": 0.927},
                {"method": "ELP-Unfolding", "psnr": 26.50, "ssim": 0.910},
                {"method": "PnP-FFDNet",    "psnr": 20.15, "ssim": 0.650},
                {"method": "GAP-TV",        "psnr": 14.81, "ssim": 0.303},
            ],
            "scenario_iii": [
                {"method": "EfficientSCI",  "psnr": 35.39, "ssim": 0.973},
                {"method": "ELP-Unfolding", "psnr": 34.09, "ssim": 0.965},
                {"method": "PnP-FFDNet",    "psnr": 29.28, "ssim": 0.910},
                {"method": "GAP-TV",        "psnr": 26.75, "ssim": 0.870},
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  SPC — Block (11 Set11 grayscale images)
    # ══════════════════════════════════════════════════════════════════════════

    "spc_block": {
        "scoring": {
            "psnr_weight": 0.40,
            "ssim_weight": 0.40,
            "consistency_weight": 0.20,
            "formula_display": "0.4 \u00d7 PSNR_norm + 0.4 \u00d7 SSIM + 0.2 \u00d7 (1 \u2212 \u2016y \u2212 H\u0302x\u0302\u2016/\u2016y\u2016)",
        },
        "spec_ranges": [
            {"name": "gain_decay_alpha", "min": 0.001, "max": 0.01, "unit": "1/measurement"},
            {"name": "noise_sigma",      "min": 0.01,  "max": 0.05, "unit": ""},
        ],
        "true_spec": {
            "gain_decay_alpha": 0.005,
            "noise_sigma": 0.03,
        },
        "noise_model": "gaussian",
        "noise_params": {"sigma": 0.03},
        "splits": {
            "pro":    {"scenes": [1, 2, 3, 4, 5, 6, 7], "count": 7},
            "hidden": {"scenes": [8, 9, 10, 11],         "count": 4},
        },
        "data_source": "datasets/SPC/Set11/",
        "data_format": "tif",
        "signal_shape": [256, 256],
        "baselines": {
            "scenario_ii": [
                {"method": "ISTA-Net",   "psnr": 27.45, "ssim": 0.760},
                {"method": "HATNet",     "psnr": 26.80, "ssim": 0.745},
                {"method": "PnP-DRUNet", "psnr": 24.10, "ssim": 0.690},
                {"method": "FISTA-TV",   "psnr": 19.02, "ssim": 0.584},
            ],
            "scenario_iii": [
                {"method": "ISTA-Net",   "psnr": 31.85, "ssim": 0.916},
                {"method": "HATNet",     "psnr": 30.98, "ssim": 0.905},
                {"method": "PnP-DRUNet", "psnr": 30.53, "ssim": 0.895},
                {"method": "FISTA-TV",   "psnr": 28.06, "ssim": 0.850},
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  SPC — Kronecker (same Set11 images, different sensing matrix structure)
    # ══════════════════════════════════════════════════════════════════════════

    "spc_kronecker": {
        "scoring": {
            "psnr_weight": 0.40,
            "ssim_weight": 0.40,
            "consistency_weight": 0.20,
            "formula_display": "0.4 \u00d7 PSNR_norm + 0.4 \u00d7 SSIM + 0.2 \u00d7 (1 \u2212 \u2016y \u2212 H\u0302x\u0302\u2016/\u2016y\u2016)",
        },
        "spec_ranges": [
            {"name": "gain_decay_alpha", "min": 0.001, "max": 0.01, "unit": "1/measurement"},
            {"name": "noise_sigma",      "min": 0.01,  "max": 0.05, "unit": ""},
        ],
        "true_spec": {
            "gain_decay_alpha": 0.005,
            "noise_sigma": 0.03,
        },
        "noise_model": "gaussian",
        "noise_params": {"sigma": 0.03},
        "splits": {
            "pro":    {"scenes": [1, 2, 3, 4, 5, 6, 7], "count": 7},
            "hidden": {"scenes": [8, 9, 10, 11],         "count": 4},
        },
        "data_source": "datasets/SPC/Set11/",
        "data_format": "tif",
        "signal_shape": [256, 256],
        "baselines": {
            "scenario_ii": [
                {"method": "ISTA-Net",   "psnr": 27.45, "ssim": 0.760},
                {"method": "HATNet",     "psnr": 26.80, "ssim": 0.745},
                {"method": "PnP-DRUNet", "psnr": 24.10, "ssim": 0.690},
                {"method": "FISTA-TV",   "psnr": 19.02, "ssim": 0.584},
            ],
            "scenario_iii": [
                {"method": "ISTA-Net",   "psnr": 31.85, "ssim": 0.916},
                {"method": "HATNet",     "psnr": 30.98, "ssim": 0.905},
                {"method": "PnP-DRUNet", "psnr": 30.53, "ssim": 0.895},
                {"method": "FISTA-TV",   "psnr": 28.06, "ssim": 0.850},
            ],
        },
    },
}

# fmt: on
