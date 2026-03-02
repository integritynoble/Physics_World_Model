"""Challenge configuration — Blind Reconstruction Challenge.

Each variant entry defines scoring weights, spec ranges (what contestants see),
per-tier mismatch configs (Public / Dev / Hidden), and baseline performance
from InverseNet Scenarios II and III.

Each tier uses DIFFERENT underlying datasets (different ground truth images)
via ``tier_data_sources``, so that knowing the public data provides no
shortcut for the blind tiers.  Tiers also use different mismatch realizations
(different true_spec values + different noise seeds).
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
            "formula_display": "0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × (1 − ‖y − Ĥx̂‖/‖y‖)",
        },
        "spec_ranges": [
            {"name": "mask_dx",          "min": 0.3,  "max": 0.7,  "unit": "px"},
            {"name": "mask_dy",          "min": 0.1,  "max": 0.5,  "unit": "px"},
            {"name": "mask_rotation",    "min": 0.0,  "max": 0.2,  "unit": "deg"},
            {"name": "dispersion_slope", "min": 1.90, "max": 2.15, "unit": "px/band"},
            {"name": "dispersion_axis",  "min": 0.0,  "max": 0.3,  "unit": "deg"},
        ],
        "noise_model": "poisson_gaussian",
        "noise_params": {"poisson_alpha": 1.0, "gaussian_sigma": 0.01},
        "scenes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "scene_count": 10,
        "tiers": {
            "public": {
                "true_spec": {
                    "mask_dx": 0.50,
                    "mask_dy": 0.30,
                    "mask_rotation": 0.10,
                    "dispersion_slope": 2.02,
                    "dispersion_axis": 0.15,
                },
                "seed": 1001,
                "visible_data": ["y", "H_ideal", "spec_ranges", "x_true", "true_spec"],
                "introduction": {
                    "summary": "Full-access development tier with all data visible.",
                    "what_you_get": "Measurements (y), ideal forward operator (H), spec ranges, ground truth (x_true), and true mismatch spec.",
                    "how_to_use": "Load HDF5 → compare reconstruction vs x_true → check consistency → iterate.",
                    "what_to_submit": "Reconstructed signals (x_hat) and corrected spec as HDF5.",
                },
                "preview_image": {"scene_idx": 1, "image_key": "gt", "alt": "Ground truth — Scene 01", "caption": "Ground truth (fully visible in Public tier)"},
            },
            "dev": {
                "true_spec": {
                    "mask_dx": 0.40,
                    "mask_dy": 0.20,
                    "mask_rotation": 0.05,
                    "dispersion_slope": 2.08,
                    "dispersion_axis": 0.10,
                },
                "seed": 2001,
                "visible_data": ["y", "H_ideal", "spec_ranges"],
                "introduction": {
                    "summary": "Blind evaluation tier — no ground truth available.",
                    "what_you_get": "Measurements (y), ideal forward operator (H), and spec ranges only.",
                    "how_to_use": "Apply your pipeline from the Public tier. Use consistency as self-check.",
                    "what_to_submit": "Reconstructed signals and corrected spec. Scored server-side.",
                },
                "preview_image": {"scene_idx": 1, "image_key": "measurement_II", "alt": "Measurement with mismatch", "caption": "Measurement only (no ground truth in Dev tier)"},
            },
            "hidden": {
                "true_spec": {
                    "mask_dx": 0.60,
                    "mask_dy": 0.40,
                    "mask_rotation": 0.15,
                    "dispersion_slope": 1.95,
                    "dispersion_axis": 0.22,
                },
                "seed": 3001,
                "visible_data": [],
                "introduction": {
                    "summary": "Fully blind server-side evaluation — no data download.",
                    "what_you_get": "No data downloadable. Algorithm runs server-side on hidden measurements.",
                    "how_to_use": "Package algorithm as Docker container / Python script. Submit via link.",
                    "what_to_submit": "Containerized algorithm accepting y + H, outputting x_hat + corrected spec.",
                },
                "preview_image": None,
            },
        },
        "data_source": "datasets/TSA_simu_data/Truth/",
        "data_format": "mat",
        "signal_shape": [256, 256, 28],
        "tier_data_sources": {
            "public":  {"path": "datasets/TSA_simu_data/Truth/", "format": "mat", "type": "experimental"},
            "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "simulated"},
            "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "simulated"},
        },
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
            "formula_display": "0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × (1 − ‖y − Ĥx̂‖/‖y‖)",
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
        "noise_model": "poisson_gaussian",
        "noise_params": {"peak_photon": 10000, "gaussian_sigma_photon": 1.0},
        "scenes": ["kobe", "traffic", "runner", "drop", "crash", "aerial"],
        "scene_count": 6,
        "tiers": {
            "public": {
                "true_spec": {
                    "mask_dx": 0.50,
                    "mask_dy": 0.30,
                    "mask_rotation": 0.10,
                    "mask_blur": 0.0,
                    "clock_offset": 0.05,
                    "gain_drift": 1.02,
                    "offset_drift": 0.002,
                },
                "seed": 1001,
                "visible_data": ["y", "H_ideal", "spec_ranges", "x_true", "true_spec"],
                "introduction": {
                    "summary": "Full-access development tier with all data visible.",
                    "what_you_get": "Measurements (y), ideal forward operator (H), spec ranges, ground truth (x_true), and true mismatch spec.",
                    "how_to_use": "Load HDF5 → compare reconstruction vs x_true → check consistency → iterate.",
                    "what_to_submit": "Reconstructed signals (x_hat) and corrected spec as HDF5.",
                },
                "preview_image": {"scene_idx": 1, "image_key": "gt", "alt": "Ground truth — Scene 01", "caption": "Ground truth (fully visible in Public tier)"},
            },
            "dev": {
                "true_spec": {
                    "mask_dx": 0.35,
                    "mask_dy": 0.20,
                    "mask_rotation": 0.08,
                    "mask_blur": 0.10,
                    "clock_offset": -0.03,
                    "gain_drift": 0.98,
                    "offset_drift": -0.01,
                },
                "seed": 2001,
                "visible_data": ["y", "H_ideal", "spec_ranges"],
                "introduction": {
                    "summary": "Blind evaluation tier — no ground truth available.",
                    "what_you_get": "Measurements (y), ideal forward operator (H), and spec ranges only.",
                    "how_to_use": "Apply your pipeline from the Public tier. Use consistency as self-check.",
                    "what_to_submit": "Reconstructed signals and corrected spec. Scored server-side.",
                },
                "preview_image": {"scene_idx": 1, "image_key": "measurement_II", "alt": "Measurement with mismatch", "caption": "Measurement only (no ground truth in Dev tier)"},
            },
            "hidden": {
                "true_spec": {
                    "mask_dx": 0.65,
                    "mask_dy": 0.40,
                    "mask_rotation": 0.22,
                    "mask_blur": 0.35,
                    "clock_offset": 0.08,
                    "gain_drift": 1.04,
                    "offset_drift": 0.015,
                },
                "seed": 3001,
                "visible_data": [],
                "introduction": {
                    "summary": "Fully blind server-side evaluation — no data download.",
                    "what_you_get": "No data downloadable. Algorithm runs server-side on hidden measurements.",
                    "how_to_use": "Package algorithm as Docker container / Python script. Submit via link.",
                    "what_to_submit": "Containerized algorithm accepting y + H, outputting x_hat + corrected spec.",
                },
                "preview_image": None,
            },
        },
        "data_source": "datasets/CACTI/simulation/",
        "data_format": "mat",
        "signal_shape": [256, 256, 8],
        "tier_data_sources": {
            "public":  {"path": "datasets/CACTI/simulation/", "format": "mat", "type": "experimental"},
            "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "simulated"},
            "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "simulated"},
        },
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
            "formula_display": "0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × (1 − ‖y − Ĥx̂‖/‖y‖)",
        },
        "spec_ranges": [
            {"name": "gain_decay_alpha", "min": 0.001, "max": 0.01, "unit": "1/measurement"},
            {"name": "noise_sigma",      "min": 0.01,  "max": 0.05, "unit": ""},
        ],
        "noise_model": "gaussian",
        "noise_params": {"sigma": 0.03},
        "scenes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "scene_count": 11,
        "tiers": {
            "public": {
                "true_spec": {
                    "gain_decay_alpha": 0.005,
                    "noise_sigma": 0.03,
                },
                "seed": 1001,
                "visible_data": ["y", "H_ideal", "spec_ranges", "x_true", "true_spec"],
                "introduction": {
                    "summary": "Full-access development tier with all data visible.",
                    "what_you_get": "Measurements (y), ideal forward operator (H), spec ranges, ground truth (x_true), and true mismatch spec.",
                    "how_to_use": "Load HDF5 → compare reconstruction vs x_true → check consistency → iterate.",
                    "what_to_submit": "Reconstructed signals (x_hat) and corrected spec as HDF5.",
                },
                "preview_image": {"scene_idx": 1, "image_key": "gt", "alt": "Ground truth — Scene 01", "caption": "Ground truth (fully visible in Public tier)"},
            },
            "dev": {
                "true_spec": {
                    "gain_decay_alpha": 0.003,
                    "noise_sigma": 0.02,
                },
                "seed": 2001,
                "visible_data": ["y", "H_ideal", "spec_ranges"],
                "introduction": {
                    "summary": "Blind evaluation tier — no ground truth available.",
                    "what_you_get": "Measurements (y), ideal forward operator (H), and spec ranges only.",
                    "how_to_use": "Apply your pipeline from the Public tier. Use consistency as self-check.",
                    "what_to_submit": "Reconstructed signals and corrected spec. Scored server-side.",
                },
                "preview_image": {"scene_idx": 1, "image_key": "measurement_II", "alt": "Measurement with mismatch", "caption": "Measurement only (no ground truth in Dev tier)"},
            },
            "hidden": {
                "true_spec": {
                    "gain_decay_alpha": 0.008,
                    "noise_sigma": 0.04,
                },
                "seed": 3001,
                "visible_data": [],
                "introduction": {
                    "summary": "Fully blind server-side evaluation — no data download.",
                    "what_you_get": "No data downloadable. Algorithm runs server-side on hidden measurements.",
                    "how_to_use": "Package algorithm as Docker container / Python script. Submit via link.",
                    "what_to_submit": "Containerized algorithm accepting y + H, outputting x_hat + corrected spec.",
                },
                "preview_image": None,
            },
        },
        "data_source": "datasets/SPC/Set11/",
        "data_format": "tif",
        "signal_shape": [256, 256],
        "tier_data_sources": {
            "public":  {"path": "datasets/SPC/Set11/", "format": "tif", "type": "experimental"},
            "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "simulated"},
            "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "simulated"},
        },
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
        "gallery_variant": "spc_block",
        "scoring": {
            "psnr_weight": 0.40,
            "ssim_weight": 0.40,
            "consistency_weight": 0.20,
            "formula_display": "0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × (1 − ‖y − Ĥx̂‖/‖y‖)",
        },
        "spec_ranges": [
            {"name": "gain_decay_alpha", "min": 0.001, "max": 0.01, "unit": "1/measurement"},
            {"name": "noise_sigma",      "min": 0.01,  "max": 0.05, "unit": ""},
        ],
        "noise_model": "gaussian",
        "noise_params": {"sigma": 0.03},
        "scenes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "scene_count": 11,
        "tiers": {
            "public": {
                "true_spec": {
                    "gain_decay_alpha": 0.005,
                    "noise_sigma": 0.03,
                },
                "seed": 1001,
                "visible_data": ["y", "H_ideal", "spec_ranges", "x_true", "true_spec"],
                "introduction": {
                    "summary": "Full-access development tier with all data visible.",
                    "what_you_get": "Measurements (y), ideal forward operator (H), spec ranges, ground truth (x_true), and true mismatch spec.",
                    "how_to_use": "Load HDF5 → compare reconstruction vs x_true → check consistency → iterate.",
                    "what_to_submit": "Reconstructed signals (x_hat) and corrected spec as HDF5.",
                },
                "preview_image": {"scene_idx": 1, "image_key": "gt", "alt": "Ground truth — Scene 01", "caption": "Ground truth (fully visible in Public tier)"},
            },
            "dev": {
                "true_spec": {
                    "gain_decay_alpha": 0.003,
                    "noise_sigma": 0.02,
                },
                "seed": 2001,
                "visible_data": ["y", "H_ideal", "spec_ranges"],
                "introduction": {
                    "summary": "Blind evaluation tier — no ground truth available.",
                    "what_you_get": "Measurements (y), ideal forward operator (H), and spec ranges only.",
                    "how_to_use": "Apply your pipeline from the Public tier. Use consistency as self-check.",
                    "what_to_submit": "Reconstructed signals and corrected spec. Scored server-side.",
                },
                "preview_image": {"scene_idx": 1, "image_key": "measurement_II", "alt": "Measurement with mismatch", "caption": "Measurement only (no ground truth in Dev tier)"},
            },
            "hidden": {
                "true_spec": {
                    "gain_decay_alpha": 0.008,
                    "noise_sigma": 0.04,
                },
                "seed": 3001,
                "visible_data": [],
                "introduction": {
                    "summary": "Fully blind server-side evaluation — no data download.",
                    "what_you_get": "No data downloadable. Algorithm runs server-side on hidden measurements.",
                    "how_to_use": "Package algorithm as Docker container / Python script. Submit via link.",
                    "what_to_submit": "Containerized algorithm accepting y + H, outputting x_hat + corrected spec.",
                },
                "preview_image": None,
            },
        },
        "data_source": "datasets/SPC/Set11/",
        "data_format": "tif",
        "signal_shape": [256, 256],
        "tier_data_sources": {
            "public":  {"path": "datasets/SPC/Set11/", "format": "tif", "type": "experimental"},
            "dev":     {"generator": "generate_synthetic_scene", "seed_offset": 77777, "type": "synthetic"},
            "hidden":  {"generator": "generate_synthetic_scene", "seed_offset": 99999, "type": "synthetic"},
        },
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
            "scenario_iv": [
                {"method": "PnP-DRUNet",      "psnr": 26.03, "ssim": 0.806},
                {"method": "FISTA-TV (tuned)", "psnr": 25.26, "ssim": 0.756},
                {"method": "FISTA-TV (paper)", "psnr": 25.07, "ssim": 0.747},
                {"method": "HATNet",           "psnr": 25.29, "ssim": 0.745},
                {"method": "ISTA-Net",         "psnr": 23.38, "ssim": 0.560},
                {"method": "PnP-BM3D",        "psnr": 19.49, "ssim": 0.533},
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  CT — 2-D Fan-Beam Sparse-View / Low-Dose
    #  Public: 11 real chest CT from LoDoPaB-CT (LIDC/IDRI, Zenodo 3384092)
    #  Dev:    20 procedural phantoms (5 background types × 4, generate_ct_gt)
    #  Hidden: 20 adversarial phantoms (metal, lesions, calcifications)
    #  Geometry: 362×362 px, D_so=800 px, 60 views (public/dev), 40–90 (hidden)
    #  Noise: Beer-Lambert + Poisson(I₀=10 000) + readout N(0,25)
    # ══════════════════════════════════════════════════════════════════════════

    "ct": {
        "scoring": {
            "psnr_weight": 0.40,
            "ssim_weight": 0.40,
            "consistency_weight": 0.20,
            "formula_display": "0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × (1 − ‖y − Ĥx̂‖/‖y‖)",
        },
        "spec_ranges": [
            {"name": "center_offset_px",    "min": -5.0, "max":  5.0, "unit": "px"},
            {"name": "angle_error_deg",     "min": -8.0, "max":  8.0, "unit": "deg"},
            {"name": "beam_hardening_beta", "min":  0.0, "max":  0.30, "unit": ""},
            {"name": "detector_tilt_deg",   "min": -3.0, "max":  3.0, "unit": "deg"},
        ],
        "noise_model": "poisson_gaussian",
        "noise_params": {"I0": 10000, "sigma_readout": 5.0},
        # Public tier has 11 scenes (test-split patients); dev/hidden have 20 each.
        # scene_count is used as the default by _factory.py; per-tier counts override it.
        "scenes": list(range(11)),
        "scene_count": 11,
        "tier_scene_counts": {"public": 11, "dev": 20, "hidden": 20},
        "tiers": {
            "public": {
                "true_spec": {
                    "center_offset_px":    1.0,
                    "angle_error_deg":     1.5,
                    "beam_hardening_beta": 0.05,
                    "detector_tilt_deg":   0.5,
                },
                "seed": 1000,
                "visible_data": ["y", "H_ideal", "spec_ranges", "x_true", "true_spec"],
                "introduction": {
                    "summary": "Full-access tier: 11 real patient CT slices from LoDoPaB-CT (LIDC/IDRI test split).",
                    "what_you_get": "Measured sinogram (y), ideal forward operator (H), spec ranges, ground truth x_true, and true mismatch spec per sample.",
                    "how_to_use": "Load ct_challenge_public.h5 → reconstruct x̂ from sinogram_measured → compare with x_true → compute consistency → iterate on mismatch correction.",
                    "what_to_submit": "Reconstructed images (x_hat) and corrected mismatch spec as HDF5.",
                },
                "preview_image": {"scene_idx": 0, "image_key": "gt", "alt": "LoDoPaB-CT chest slice", "caption": "Real chest CT (LoDoPaB-CT, LIDC/IDRI) — Ground truth visible in Public tier"},
            },
            "dev": {
                "true_spec": {
                    "center_offset_px":    2.0,
                    "angle_error_deg":     3.0,
                    "beam_hardening_beta": 0.08,
                    "detector_tilt_deg":   1.0,
                },
                "seed": 7000,
                "visible_data": ["y", "H_ideal", "spec_ranges"],
                "introduction": {
                    "summary": "Blind evaluation: 20 real patient CT slices from LoDoPaB-CT (validation split, patients 0–63).",
                    "what_you_get": "Measured sinogram (y), ideal forward operator (H), and spec ranges. No ground truth.",
                    "how_to_use": "Apply your pipeline from Public tier. Self-check via consistency metric. Ground truth scored server-side.",
                    "what_to_submit": "Reconstructed images and corrected mismatch spec. Scored server-side.",
                },
                "preview_image": {"scene_idx": 0, "image_key": "measurement_II", "alt": "Measured sinogram", "caption": "Measured sinogram only (no ground truth in Dev tier)"},
            },
            "hidden": {
                "true_spec": {
                    "center_offset_px":    4.0,
                    "angle_error_deg":     6.0,
                    "beam_hardening_beta": 0.22,
                    "detector_tilt_deg":   2.5,
                },
                "seed": 9000,
                "visible_data": [],
                "introduction": {
                    "summary": "Fully blind: 20 real LoDoPaB-CT slices (validation split, patients 64–127) with adversarial modifications (metal inserts, lesions, calcifications).",
                    "what_you_get": "No data download. Algorithm runs server-side on hidden measurements.",
                    "how_to_use": "Package algorithm as Docker container / Python script accepting y + H, outputting x_hat + corrected spec.",
                    "what_to_submit": "Containerized algorithm. Scored server-side against adversarial phantoms.",
                },
                "preview_image": None,
            },
        },
        "data_source": "datasets/benchmark/ct/",
        "data_format": "hdf5",
        "signal_shape": [362, 362],
        "tier_data_sources": {
            "public": {"path": "datasets/benchmark/ct/public/ct_challenge_public.h5",  "format": "hdf5", "type": "real"},
            "dev":    {"path": "datasets/benchmark/ct/dev/ct_challenge_dev.h5",         "format": "hdf5", "type": "real"},
            "hidden": {"path": "datasets/benchmark/ct/hidden/ct_challenge_hidden.h5",   "format": "hdf5", "type": "real_adversarial"},
        },
        # Scenario I: blind reconstruction — algorithm uses H_nom, measured noisy sinogram.
        # Measured on actual LoDoPaB-CT challenge HDF5s (public tier, 11 scenes).
        # Run: scripts/run_ct_benchmark.py + scripts/modal_run_ct_benchmark.py · 2026-03-01
        #
        # Scenario II: mismatched operator — algorithm uses H_ideal, true-spec data.
        # Scenario III: corrected operator — true mismatch params known (oracle).
        # Numbers from published CT literature calibrated to our geometry.
        "baselines": {
            "scenario_i": [
                {"method": "FBP",           "psnr": 21.84, "ssim": 0.382, "score": 0.440, "tier": "public"},
                {"method": "PnP-ADMM",      "psnr": 23.21, "ssim": 0.621, "score": 0.556, "tier": "public"},
                {"method": "PnP-DRUNet",    "psnr": 24.51, "ssim": 0.707, "score": 0.610, "tier": "public", "device": "T4"},
            ],
            "scenario_ii": [
                {"method": "FBP",                 "psnr": 23.14, "ssim": 0.641},
                {"method": "PnP-ADMM",            "psnr": 25.83, "ssim": 0.730},
                {"method": "FBPConvNet",           "psnr": 24.95, "ssim": 0.712},
                {"method": "Learned Primal-Dual",  "psnr": 27.35, "ssim": 0.780},
                {"method": "DuDoTrans",            "psnr": 26.80, "ssim": 0.762},
                {"method": "DOLCE",                "psnr": 28.10, "ssim": 0.805},
            ],
            "scenario_iii": [
                {"method": "FBP",                 "psnr": 26.10, "ssim": 0.762},
                {"method": "PnP-ADMM",            "psnr": 29.72, "ssim": 0.855},
                {"method": "FBPConvNet",           "psnr": 30.40, "ssim": 0.872},
                {"method": "Learned Primal-Dual",  "psnr": 34.15, "ssim": 0.932},
                {"method": "DuDoTrans",            "psnr": 35.42, "ssim": 0.948},
                {"method": "DOLCE",                "psnr": 36.80, "ssim": 0.961},
            ],
        },
    },
}

# fmt: on

# ══════════════════════════════════════════════════════════════════════════════
#  Auto-generation helpers — derive challenge configs from mismatch_params
# ══════════════════════════════════════════════════════════════════════════════

import re


def _extract_unit(description: str) -> str:
    """Extract unit from a mismatch_param description like 'Mask shift (pixels)' → 'pixels'."""
    m = re.search(r"\(([^)]+)\)\s*$", description)
    if m:
        unit = m.group(1)
        # Normalize common patterns
        return unit.replace(" ", "")
    return ""


def _derive_spec_ranges(mismatch_params: list[dict]) -> list[dict]:
    """Convert variant mismatch_params into challenge spec_ranges.

    Each range is wider than the actual mismatch (±1.5×delta) to add
    uncertainty for the contestant.
    """
    ranges = []
    for mp in mismatch_params:
        nominal = mp["nominal"]
        perturbed = mp["perturbed"]
        delta = abs(perturbed - nominal)
        # Edge case: if delta ≈ 0, use ±10% of nominal or ±0.1
        if delta < 1e-12:
            delta = max(abs(nominal) * 0.1, 0.1)
        margin = 1.5 * delta
        lo = nominal - margin
        hi = nominal + margin
        # Round to reasonable precision
        lo = round(lo, 6)
        hi = round(hi, 6)
        ranges.append({
            "name": mp["name"],
            "min": lo,
            "max": hi,
            "unit": _extract_unit(mp.get("description", "")),
        })
    return ranges


def _derive_true_spec_for_tier(
    mismatch_params: list[dict],
    tier: str,
) -> dict[str, float]:
    """Generate tier-specific true mismatch values by interpolating.

    - public: 50% toward perturbed (moderate, with ground truth for validation)
    - dev:    30% toward perturbed (milder, blind evaluation)
    - hidden: 80% toward perturbed (severe, tests robustness)
    """
    fractions = {"public": 0.50, "dev": 0.30, "hidden": 0.80}
    frac = fractions[tier]
    spec = {}
    for mp in mismatch_params:
        nominal = mp["nominal"]
        perturbed = mp["perturbed"]
        value = nominal + frac * (perturbed - nominal)
        spec[mp["name"]] = round(value, 6)
    return spec


def _derive_tier_spec_ranges(
    base_spec_ranges: list[dict],
    true_spec: dict[str, float],
) -> list[dict]:
    """Shift spec_ranges to center on a tier's true_spec values.

    Keeps the same range width but re-centers on the true value.
    """
    result = []
    for sr in base_spec_ranges:
        name = sr["name"]
        original_width = sr["max"] - sr["min"]
        half_width = original_width / 2
        center = true_spec.get(name, (sr["min"] + sr["max"]) / 2)
        result.append({
            "name": name,
            "min": round(center - half_width, 6),
            "max": round(center + half_width, 6),
            "unit": sr["unit"],
        })
    return result


# ── Backfill per-tier spec_ranges for hand-crafted configs ────────────────────

for _cfg in CHALLENGE_CONFIG.values():
    _base_ranges = _cfg.get("spec_ranges", [])
    for _tier in _cfg["tiers"].values():
        if "spec_ranges" not in _tier and "true_spec" in _tier:
            _tier["spec_ranges"] = _derive_tier_spec_ranges(_base_ranges, _tier["true_spec"])


# ── Category defaults ─────────────────────────────────────────────────────────

_CATEGORY_NOISE_MODEL: dict[str, str] = {
    "compressive":                "poisson_gaussian",
    "medical":                    "poisson",
    "medical_ultrasound":         "speckle",
    "coherent":                   "gaussian",
    "microscopy":                 "poisson_gaussian",
    "electron_microscopy":        "poisson",
    "clinical_optics":            "gaussian",
    "computational":              "gaussian",
    "computational_photography":  "poisson_gaussian",
    "neural_rendering":           "gaussian",
    "depth_imaging":              "gaussian",
    "remote_sensing":             "speckle",
    "particle_imaging":           "poisson",
    "scanning_probe":             "gaussian",
    "industrial_inspection":      "gaussian",
    "spectroscopy":               "gaussian",
    "astronomy":                  "poisson",
    "ultrafast":                  "poisson_gaussian",
    "quantum":                    "poisson",
    "experimental_science":       "gaussian",
    "scientific_instrumentation": "gaussian",
    "multi_modal_fusion":         "gaussian",
}

_CATEGORY_SIGNAL_SHAPE: dict[str, list[int]] = {
    "compressive":                [256, 256],
    "medical":                    [128, 128, 64],
    "medical_ultrasound":         [256, 256],
    "coherent":                   [256, 256],
    "microscopy":                 [256, 256],
    "electron_microscopy":        [512, 512],
    "clinical_optics":            [256, 256],
    "computational":              [128, 128],
    "computational_photography":  [256, 256],
    "neural_rendering":           [800, 800],
    "depth_imaging":              [256, 256],
    "remote_sensing":             [512, 512],
    "particle_imaging":           [128, 128, 64],
    "scanning_probe":             [256, 256],
    "industrial_inspection":      [256, 256],
    "spectroscopy":               [256, 256],
    "astronomy":                  [512, 512],
    "ultrafast":                  [256, 256],
    "quantum":                    [128, 128],
    "experimental_science":       [256, 256],
    "scientific_instrumentation": [256, 256],
    "multi_modal_fusion":         [256, 256],
}

_CATEGORY_SCENE_COUNT: dict[str, int] = {
    "compressive":                5,
    "medical":                    3,
    "medical_ultrasound":         3,
    "coherent":                   5,
    "microscopy":                 5,
    "electron_microscopy":        3,
    "clinical_optics":            3,
    "computational":              5,
    "computational_photography":  5,
    "neural_rendering":           5,
    "depth_imaging":              5,
    "remote_sensing":             3,
    "particle_imaging":           3,
    "scanning_probe":             3,
    "industrial_inspection":      3,
    "spectroscopy":               5,
    "astronomy":                  3,
    "ultrafast":                  5,
    "quantum":                    3,
    "experimental_science":       5,
    "scientific_instrumentation": 5,
    "multi_modal_fusion":         3,
}

# ── Standard templates ────────────────────────────────────────────────────────

_STANDARD_SCORING: dict = {
    "psnr_weight": 0.40,
    "ssim_weight": 0.40,
    "consistency_weight": 0.20,
    "formula_display": "0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × (1 − ‖y − Ĥx̂‖/‖y‖)",
}

_TIER_TEMPLATES: dict[str, dict] = {
    "public": {
        "seed": 1001,
        "visible_data": ["y", "H_ideal", "spec_ranges", "x_true", "true_spec"],
        "introduction": {
            "summary": "Full-access development tier with all data visible.",
            "what_you_get": "Measurements (y), ideal forward operator (H), spec ranges, ground truth (x_true), and true mismatch spec.",
            "how_to_use": "Load HDF5 → compare reconstruction vs x_true → check consistency → iterate.",
            "what_to_submit": "Reconstructed signals (x_hat) and corrected spec as HDF5.",
        },
        "preview_image": {"scene_idx": 1, "image_key": "gt", "alt": "Ground truth — Scene 01", "caption": "Ground truth (fully visible in Public tier)"},
    },
    "dev": {
        "seed": 2001,
        "visible_data": ["y", "H_ideal", "spec_ranges"],
        "introduction": {
            "summary": "Blind evaluation tier — no ground truth available.",
            "what_you_get": "Measurements (y), ideal forward operator (H), and spec ranges only.",
            "how_to_use": "Apply your pipeline from the Public tier. Use consistency as self-check.",
            "what_to_submit": "Reconstructed signals and corrected spec. Scored server-side.",
        },
        "preview_image": {"scene_idx": 1, "image_key": "measurement_II", "alt": "Measurement with mismatch", "caption": "Measurement only (no ground truth in Dev tier)"},
    },
    "hidden": {
        "seed": 3001,
        "visible_data": [],
        "introduction": {
            "summary": "Fully blind server-side evaluation — no data download.",
            "what_you_get": "No data downloadable. Algorithm runs server-side on hidden measurements.",
            "how_to_use": "Package algorithm as Docker container / Python script. Submit via link.",
            "what_to_submit": "Containerized algorithm accepting y + H, outputting x_hat + corrected spec.",
        },
        "preview_image": None,
    },
}


# ── Per-tier data source mapping for auto-generated variants ─────────────────
# Each category maps to a dict of tier → data source config.
# Public tier may use real/web data; Dev and Hidden ALWAYS use simulated data
# (generator-based) so they remain private to the server and cannot be found
# by contestants online.
# Fallback chain: path → registry_id → generator → default with seed offset.

_CATEGORY_TIER_DATA_SOURCES: dict[str, dict[str, dict]] = {
    "compressive": {
        "public":  {"registry_id": "indian_pines_hs", "type": "web"},
        "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "simulated"},
        "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "simulated"},
    },
    "microscopy": {
        "public":  {"registry_id": "bsd68_microscopy", "type": "web"},
        "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "generated"},
    },
    "medical": {
        "public":  {"registry_id": "lodopab_ct_sample", "type": "web"},
        "dev":     {"generator": "generate_medical_phantom", "seed_offset": 10000, "type": "simulated"},
        "hidden":  {"generator": "generate_medical_phantom", "seed_offset": 20000, "type": "simulated"},
    },
    "medical_ultrasound": {
        "public":  {"generator": "generate_medical_phantom", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_medical_phantom", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_medical_phantom", "seed_offset": 20000, "type": "generated"},
    },
    "coherent": {
        "public":  {"generator": "generate_resolution_target", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_resolution_target", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_resolution_target", "seed_offset": 20000, "type": "generated"},
    },
    "electron_microscopy": {
        "public":  {"generator": "generate_em_phantom", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_em_phantom", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_em_phantom", "seed_offset": 20000, "type": "generated"},
    },
    "clinical_optics": {
        "public":  {"generator": "generate_oct_phantom", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_oct_phantom", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_oct_phantom", "seed_offset": 20000, "type": "generated"},
    },
    "computational": {
        "public":  {"generator": "generate_test_scene", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "generated"},
    },
    "computational_photography": {
        "public":  {"generator": "generate_test_scene", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "generated"},
    },
    "neural_rendering": {
        "public":  {"generator": "generate_test_scene", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "generated"},
    },
    "depth_imaging": {
        "public":  {"generator": "generate_depth_map", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_depth_map", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_depth_map", "seed_offset": 20000, "type": "generated"},
    },
    "remote_sensing": {
        "public":  {"registry_id": "kennedy_space_center_hs", "type": "web"},
        "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "simulated"},
        "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "simulated"},
    },
    "particle_imaging": {
        "public":  {"generator": "generate_medical_phantom", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_medical_phantom", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_medical_phantom", "seed_offset": 20000, "type": "generated"},
    },
    "scanning_probe": {
        "public":  {"generator": "generate_surface", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_surface", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_surface", "seed_offset": 20000, "type": "generated"},
    },
    "industrial_inspection": {
        "public":  {"generator": "generate_ndt_phantom", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_ndt_phantom", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_ndt_phantom", "seed_offset": 20000, "type": "generated"},
    },
    "spectroscopy": {
        "public":  {"registry_id": "pavia_university_hs", "type": "web"},
        "dev":     {"generator": "generate_elemental_map", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_elemental_map", "seed_offset": 20000, "type": "generated"},
    },
    "astronomy": {
        "public":  {"generator": "generate_star_field", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_star_field", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_star_field", "seed_offset": 20000, "type": "generated"},
    },
    "ultrafast": {
        "public":  {"generator": "generate_test_scene", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "generated"},
    },
    "quantum": {
        "public":  {"generator": "generate_test_scene", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "generated"},
    },
    "experimental_science": {
        "public":  {"generator": "generate_test_scene", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "generated"},
    },
    "scientific_instrumentation": {
        "public":  {"generator": "generate_diffraction_pattern", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_diffraction_pattern", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_diffraction_pattern", "seed_offset": 20000, "type": "generated"},
    },
    "multi_modal_fusion": {
        "public":  {"generator": "generate_test_scene", "seed_offset": 0, "type": "generated"},
        "dev":     {"generator": "generate_test_scene", "seed_offset": 10000, "type": "generated"},
        "hidden":  {"generator": "generate_test_scene", "seed_offset": 20000, "type": "generated"},
    },
}


# ── Main auto-generation function ─────────────────────────────────────────────


def generate_challenge_config(
    key: str,
    mismatch_params: list[dict],
    category: str,
    dataset_config: dict | None = None,
    baselines: dict | None = None,
) -> dict | None:
    """Auto-generate a challenge config from a variant's mismatch_params.

    Returns None if mismatch_params is empty (nothing to challenge on).

    Parameters
    ----------
    baselines : dict, optional
        Pre-computed baselines with keys "scenario_ii" and "scenario_iii",
        each a list of {"method", "psnr", "ssim"} dicts.  If provided,
        these replace the empty default baselines.
    """
    if not mismatch_params:
        return None

    scene_count = _CATEGORY_SCENE_COUNT.get(category, 5)
    noise_model = _CATEGORY_NOISE_MODEL.get(category, "gaussian")
    signal_shape = _CATEGORY_SIGNAL_SHAPE.get(category, [256, 256])

    spec_ranges = _derive_spec_ranges(mismatch_params)

    tiers = {}
    for tier_name, template in _TIER_TEMPLATES.items():
        true_spec = _derive_true_spec_for_tier(mismatch_params, tier_name)
        tiers[tier_name] = {
            "true_spec": true_spec,
            "spec_ranges": _derive_tier_spec_ranges(spec_ranges, true_spec),
            "seed": template["seed"],
            "visible_data": list(template["visible_data"]),
            "introduction": dict(template["introduction"]),
            "preview_image": dict(template["preview_image"]) if template["preview_image"] else None,
        }

    default_baselines = {
        "scenario_ii": [],
        "scenario_iii": [],
    }

    # Look up per-tier data sources from category mapping
    tier_data_sources = _CATEGORY_TIER_DATA_SOURCES.get(category)

    result = {
        "_auto_generated": True,
        "scoring": dict(_STANDARD_SCORING),
        "spec_ranges": spec_ranges,
        "noise_model": noise_model,
        "noise_params": {},
        "scenes": list(range(1, scene_count + 1)),
        "scene_count": scene_count,
        "tiers": tiers,
        "data_format": "hdf5",
        "signal_shape": list(signal_shape),
        "baselines": baselines if baselines else default_baselines,
    }

    if tier_data_sources:
        result["tier_data_sources"] = tier_data_sources

    return result
