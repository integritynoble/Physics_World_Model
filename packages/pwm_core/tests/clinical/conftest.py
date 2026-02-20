"""Shared pytest fixtures for the Clinical CT QC Copilot test suite.

Provides synthetic ACR CT phantom volumes with known ground-truth geometry
and HU values, as well as sample configuration dicts and threshold YAML
files used across multiple test modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Phantom geometry constants
# ---------------------------------------------------------------------------

PHANTOM_MATRIX = 256
PHANTOM_SLICES = 20
PHANTOM_DIAMETER_MM = 200.0
PIXEL_SPACING_MM = (0.78125, 0.78125)  # 200 mm FOV / 256 pixels

# Pixel-domain radius/offsets derived from physical dimensions
_PIX_PER_MM = 1.0 / PIXEL_SPACING_MM[0]  # ~1.28 px/mm
PHANTOM_RADIUS_PX = (PHANTOM_DIAMETER_MM / 2.0) * _PIX_PER_MM  # ~128 px
INSERT_OFFSET_PX = 50.0 * _PIX_PER_MM  # 50 mm from centre
INSERT_RADIUS_PX = 5.0 * _PIX_PER_MM  # 5 mm insert radius

# True centre of the phantom in the image matrix
CENTER_Y = PHANTOM_MATRIX // 2  # 128
CENTER_X = PHANTOM_MATRIX // 2  # 128

# Insert definitions: (angle_deg, nominal_hu)
# Angles follow the ACR CasePack convention: clockwise from 12-o'clock.
#   0 deg = 12 o'clock, 90 deg = 3 o'clock, 180 deg = 6 o'clock, 270 deg = 9 o'clock.
# This matches compute_ct_number_insert:
#   insert_row = center[0] - offset * cos(angle_rad)
#   insert_col = center[1] + offset * sin(angle_rad)
INSERT_DEFS: dict[str, tuple[float, float]] = {
    "bone":         (90.0,   955.0),
    "air":          (180.0, -1000.0),
    "acrylic":      (270.0,  120.0),
    "polyethylene": (0.0,    -95.0),
}


# ---------------------------------------------------------------------------
# Helper: build a 2-D circular mask
# ---------------------------------------------------------------------------

def _circle_mask(
    shape: tuple[int, int],
    cy: float,
    cx: float,
    radius: float,
) -> np.ndarray:
    """Return a boolean mask for a filled circle in a 2-D array."""
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pixel_spacing() -> tuple[float, float]:
    """Return the pixel spacing (row, col) in mm for the synthetic phantom.

    Typical for a 256x256 reconstruction of a ~200 mm FOV.
    """
    return PIXEL_SPACING_MM


@pytest.fixture()
def synthetic_acr_phantom() -> np.ndarray:
    """Generate a 256x256x20 synthetic ACR CT phantom volume.

    The phantom consists of:
      - Background air at -1000 HU
      - A circular water body (200 mm diameter) at 0 HU
      - Four material inserts at known angular positions:
            bone (955 HU), air (-1000 HU), acrylic (120 HU),
            polyethylene (-95 HU)
      - Mild additive Gaussian noise (std = 5 HU)

    Returns
    -------
    np.ndarray
        Volume of shape ``(20, 256, 256)`` in Hounsfield Units.
    """
    rng = np.random.default_rng(seed=42)
    volume = np.full(
        (PHANTOM_SLICES, PHANTOM_MATRIX, PHANTOM_MATRIX),
        -1000.0,
        dtype=np.float64,
    )

    # Water body
    body_mask = _circle_mask(
        (PHANTOM_MATRIX, PHANTOM_MATRIX),
        CENTER_Y, CENTER_X, PHANTOM_RADIUS_PX,
    )
    for z in range(PHANTOM_SLICES):
        volume[z][body_mask] = 0.0

    # Material inserts (placed in every slice for simplicity)
    # CW from 12-o'clock: row = center - offset*cos(angle), col = center + offset*sin(angle)
    for _name, (angle_deg, hu_val) in INSERT_DEFS.items():
        angle_rad = np.deg2rad(angle_deg)
        iy = CENTER_Y - INSERT_OFFSET_PX * np.cos(angle_rad)
        ix = CENTER_X + INSERT_OFFSET_PX * np.sin(angle_rad)
        ins_mask = _circle_mask(
            (PHANTOM_MATRIX, PHANTOM_MATRIX), iy, ix, INSERT_RADIUS_PX,
        )
        for z in range(PHANTOM_SLICES):
            volume[z][ins_mask] = hu_val

    # Add Gaussian noise
    noise = rng.normal(loc=0.0, scale=5.0, size=volume.shape)
    volume += noise

    return volume


@pytest.fixture()
def synthetic_acr_phantom_with_artifacts(
    synthetic_acr_phantom: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return a dict of phantom volumes with and without artifacts.

    Keys
    ----
    ``"clean"``
        The original synthetic phantom (reference).
    ``"ring"``
        Phantom corrupted by a ring artifact: a sinusoidal azimuthal
        modulation at a fixed radius band, simulating detector gain drift.
    ``"cupping"``
        Phantom corrupted by a cupping artifact: a parabolic radial HU
        depression strongest at the centre, simulating beam-hardening.
    ``"streak"``
        Phantom corrupted by streak artifacts: bright directional lines
        through the phantom, simulating a dead detector channel.
    """
    clean = synthetic_acr_phantom.copy()

    # ------------------------------------------------------------------
    # Ring artifact
    # ------------------------------------------------------------------
    ring = clean.copy()
    yy, xx = np.mgrid[:PHANTOM_MATRIX, :PHANTOM_MATRIX]
    dy = yy - CENTER_Y
    dx = xx - CENTER_X
    r = np.sqrt(dy ** 2 + dx ** 2)
    theta = np.arctan2(dy, dx)

    # Sinusoidal variation in a narrow radial band (radius 30-50 px)
    # placed well inside the phantom and away from inserts at ~64 px
    ring_band = (r >= 30) & (r <= 50)
    ring_pattern = 100.0 * np.sin(12.0 * theta)  # amplitude 100 HU
    for z in range(PHANTOM_SLICES):
        ring[z][ring_band] += ring_pattern[ring_band]

    # ------------------------------------------------------------------
    # Cupping artifact
    # ------------------------------------------------------------------
    cupping = clean.copy()
    r_norm = r / PHANTOM_RADIUS_PX  # 0 at centre, 1 at edge
    body_mask = _circle_mask(
        (PHANTOM_MATRIX, PHANTOM_MATRIX),
        CENTER_Y, CENTER_X, PHANTOM_RADIUS_PX,
    )
    # Parabolic depression: -20 HU at centre, 0 at edge
    cupping_profile = -20.0 * (1.0 - r_norm ** 2)
    for z in range(PHANTOM_SLICES):
        cupping[z][body_mask] += cupping_profile[body_mask]

    # ------------------------------------------------------------------
    # Streak artifact
    # ------------------------------------------------------------------
    streak = clean.copy()
    # Bright horizontal lines through the centre (width 3 pixels)
    streak_rows = [CENTER_Y - 1, CENTER_Y, CENTER_Y + 1]
    for z in range(PHANTOM_SLICES):
        for row in streak_rows:
            streak[z, row, :] += 200.0  # +200 HU streak

    return {
        "clean": clean,
        "ring": ring,
        "cupping": cupping,
        "streak": streak,
    }


@pytest.fixture()
def sample_casepack_config() -> dict[str, Any]:
    """Return a minimal ACR CT CasePack configuration dict.

    This mirrors the structure of ``acr_ct.yaml`` with the fields most
    commonly accessed by the metric and threshold pipelines.
    """
    return {
        "casepack": {
            "id": "acr_ct_v1.0",
            "name": "ACR CT Phantom QC",
            "version": "1.0.0",
            "min_pwm_version": "0.4.0",
            "phantom_type": "ACR CT 464",
            "series_selection": {
                "rules": [
                    {
                        "name": "module_1_axial",
                        "match": {
                            "series_description_contains": ["ACR", "Module 1"],
                            "slice_thickness_range": [4.0, 6.0],
                            "image_type_contains": "AXIAL",
                        },
                    },
                ],
            },
            "roi_definitions": {
                "water_roi": {
                    "shape": "circle",
                    "center_method": "phantom_center_auto",
                    "radius_mm": 20.0,
                },
                "peripheral_rois": {
                    "shape": "circle",
                    "count": 4,
                    "radius_mm": 10.0,
                    "offset_from_center_mm": 60.0,
                    "positions": [
                        "12_oclock", "3_oclock", "6_oclock", "9_oclock",
                    ],
                },
                "insert_rois": {
                    "bone": {
                        "offset_angle_deg": 90,
                        "offset_radius_mm": 50,
                        "expected_hu": 955,
                    },
                    "air": {
                        "offset_angle_deg": 180,
                        "offset_radius_mm": 50,
                        "expected_hu": -1000,
                    },
                    "acrylic": {
                        "offset_angle_deg": 270,
                        "offset_radius_mm": 50,
                        "expected_hu": 120,
                    },
                    "polyethylene": {
                        "offset_angle_deg": 0,
                        "offset_radius_mm": 50,
                        "expected_hu": -95,
                    },
                },
            },
            "metric_set": [
                "ct_number_water",
                "ct_number_bone",
                "ct_number_air",
                "ct_number_acrylic",
                "ct_number_polyethylene",
                "geometric_accuracy",
                "slice_thickness",
                "uniformity",
                "noise_std",
                "low_contrast_detectability",
                "artifact_evaluation",
                "spatial_resolution",
            ],
            "threshold_set": "acr_ct_thresholds",
            "report_template": "ct_qa_report",
        },
    }


@pytest.fixture()
def sample_metrics() -> dict[str, float]:
    """Return a sample QA metrics dict with typical passing values.

    These values are representative of what would be measured on a
    well-functioning scanner with an ACR CT 464 phantom.
    """
    return {
        "ct_number_water": 0.8,
        "ct_number_bone": 952.0,
        "ct_number_air": -998.0,
        "ct_number_acrylic": 118.5,
        "ct_number_polyethylene": -96.2,
        "uniformity": 2.1,
        "noise_std": 7.8,
        "geometric_accuracy": 200.3,
        "slice_thickness": 5.1,
        "low_contrast_detectability": 4,
        "artifact_evaluation": 0.0,
        "spatial_resolution": 6,
    }


@pytest.fixture()
def sample_baseline_metrics() -> dict[str, dict[str, Any]]:
    """Return baseline metrics dict in CommissioningBundle format.

    Each entry maps metric name to ``{"value": ..., "unit": ...}``.
    """
    return {
        "ct_number_water": {"value": 0.5, "unit": "HU"},
        "ct_number_bone": {"value": 954.0, "unit": "HU"},
        "ct_number_air": {"value": -999.0, "unit": "HU"},
        "ct_number_acrylic": {"value": 119.0, "unit": "HU"},
        "ct_number_polyethylene": {"value": -95.5, "unit": "HU"},
        "uniformity": {"value": 2.0, "unit": "HU"},
        "noise_std": {"value": 7.5, "unit": "HU"},
        "geometric_accuracy": {"value": 200.0, "unit": "mm"},
        "slice_thickness": {"value": 5.0, "unit": "mm"},
        "low_contrast_detectability": {"value": 4, "unit": "targets"},
        "artifact_evaluation": {"value": 0.0, "unit": "HU"},
        "spatial_resolution": {"value": 6, "unit": "lp/cm"},
    }


@pytest.fixture()
def sample_threshold_yaml(tmp_path: Path) -> Path:
    """Create a temporary 4-layer threshold YAML file and return its path.

    The four layers (in ascending priority) are:

    1. **standard** -- ACR programme defaults.
    2. **scanner_model** -- Scanner-specific overrides (e.g. Siemens
       SOMATOM Force has tighter noise tolerances).
    3. **protocol** -- Protocol-specific expected baselines.
    4. **site** -- Site-level overrides (tightest).
    """
    content = """\
# 4-layer threshold configuration for ACR CT QC
layers:
  standard:
    ct_number_water:
      range: [-7.0, 7.0]
      unit: HU
    ct_number_bone:
      range: [850, 970]
      unit: HU
    ct_number_air:
      range: [-1005, -970]
      unit: HU
    ct_number_acrylic:
      range: [110, 135]
      unit: HU
    ct_number_polyethylene:
      range: [-107, -84]
      unit: HU
    uniformity:
      range: [0, 5.0]
      unit: HU
    noise_std:
      range: [0, 12.0]
      unit: HU
    geometric_accuracy:
      range: [198.0, 202.0]
      unit: mm
    slice_thickness:
      range: [4.5, 5.5]
      unit: mm

  scanner_model:
    "Siemens SOMATOM Force":
      noise_std:
        range: [0, 9.0]
        unit: HU
      uniformity:
        range: [0, 3.5]
        unit: HU

  protocol:
    "ACR CT Adult Abdomen 120kVp":
      ct_number_water:
        expected_baseline: 0.5
        tolerance_from_baseline_sigma: 2.0

  site:
    "Main Hospital CT-001":
      noise_std:
        range: [0, 8.0]
        unit: HU
      uniformity:
        range: [0, 3.0]
        unit: HU
"""
    yaml_path = tmp_path / "thresholds.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


@pytest.fixture()
def sample_mismatch_yaml(tmp_path: Path) -> Path:
    """Create a temporary mismatch library YAML and return its path.

    The mismatch library encodes known cause-effect relationships between
    observable QC metric deviations and underlying scanner/physics issues.
    Each entry maps an observable feature signature to one or more
    candidate root causes (hypotheses) with associated likelihoods.
    """
    content = """\
# Mismatch library: feature-signature -> root-cause hypotheses
mismatch_library:
  ring_artifact:
    features:
      ring_index: "> 0.3"
    hypotheses:
      - name: detector_gain_drift
        likelihood: 0.7
        description: "One or more detector elements have drifted in gain."
      - name: center_of_rotation
        likelihood: 0.3
        description: "Center-of-rotation offset produces ring-like artifacts."

  cupping_artifact:
    features:
      cupping_index: "> 3.0"
    hypotheses:
      - name: scatter_fraction_drift
        likelihood: 0.5
        description: "Scatter correction is under-estimating scatter fraction."
      - name: beam_hardening_residual
        likelihood: 0.5
        description: "Beam hardening correction is insufficient."

  streak_artifact:
    features:
      streak_index: "> 0.3"
    hypotheses:
      - name: detector_dead_channel
        likelihood: 0.8
        description: "Dead or intermittent detector channel."
      - name: helical_interpolation
        likelihood: 0.2
        description: "Helical interpolation artifact at high pitch."

  hu_drift:
    features:
      hu_drift_index: "> 0.05"
    hypotheses:
      - name: hu_calibration_slope
        likelihood: 0.6
        description: "HU calibration slope has drifted since commissioning."
      - name: tube_aging
        likelihood: 0.4
        description: "X-ray tube spectral shift due to target pitting."

  noise_increase:
    features:
      noise_ratio: "> 1.5"
    hypotheses:
      - name: noise_floor_increase
        likelihood: 0.7
        description: "Detector electronics noise floor has increased."
      - name: tube_output_drop
        likelihood: 0.3
        description: "Tube output has dropped, reducing signal-to-noise."
"""
    yaml_path = tmp_path / "mismatch_library.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path
