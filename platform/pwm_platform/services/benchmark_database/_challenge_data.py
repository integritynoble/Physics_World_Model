"""Challenge configuration — Blind Reconstruction Challenge (Benchmark New).

Each variant entry defines scoring weights, spec ranges (what contestants see),
per-tier mismatch configs (Public / Dev / Hidden), and baseline performance
from InverseNet Scenarios II and III.

All three tiers use ALL scenes but with different mismatch realizations
(different true_spec values + different noise seeds), preventing cheating
across tiers.
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
        "noise_params": {"poisson_alpha": 1.0, "gaussian_sigma": 0.01},
        "scenes": ["kobe", "traffic", "runner", "drop", "crash", "aerial"],
        "scene_count": 6,
        "tiers": {
            "public": {
                "true_spec": {
                    "mask_dx": 0.50,
                    "mask_dy": 0.30,
                    "mask_rotation": 0.15,
                    "mask_blur": 0.20,
                    "clock_offset": 0.05,
                    "gain_drift": 1.02,
                    "offset_drift": 0.01,
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

    return {
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
