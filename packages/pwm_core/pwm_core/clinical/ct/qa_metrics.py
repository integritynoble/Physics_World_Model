"""CT QA metric computation from reconstructed DICOM image volumes.

Metric-first approach: all metrics are computed directly from pixel data
and DICOM metadata. No forward model is required.

Each metric is a standalone function with documented formula and ACR reference.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MetricResult(BaseModel):
    """Result of a single QA metric computation.

    Attributes
    ----------
    name : str
        Human-readable metric name (e.g. ``"CT Number - Water"``).
    value : float
        Computed metric value in the declared ``unit``.
    unit : str
        Physical or logical unit (e.g. ``"HU"``, ``"mm"``, ``"lp/cm"``).
    status : str
        One of ``"computed"``, ``"error"``, ``"skipped"``.
    derivation : dict
        Traceability record describing how the value was computed
        (formula, intermediate values, thresholds applied).
    roi_info : dict
        Spatial coordinates and dimensions of the ROI(s) used.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    unit: str
    status: str = "computed"
    derivation: Dict[str, Any] = Field(default_factory=dict)
    roi_info: Dict[str, Any] = Field(default_factory=dict)


class QAMetricsReport(BaseModel):
    """Complete QA metrics report for a single CT scanner session.

    Attributes
    ----------
    scanner_id : str
        Station name or scanner identifier.
    date : str
        Study date string (YYYYMMDD).
    casepack_id : str
        CasePack identifier used for the analysis.
    metrics : dict[str, MetricResult]
        Mapping from metric key to its result.
    computation_time_sec : float
        Wall-clock time in seconds for all metrics.
    """

    model_config = ConfigDict(extra="forbid")

    scanner_id: str
    date: str
    casepack_id: str
    metrics: Dict[str, MetricResult] = Field(default_factory=dict)
    computation_time_sec: float = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _mm_to_pixels(mm: float, pixel_spacing: float) -> int:
    """Convert a millimetre measurement to the nearest whole-pixel count.

    Parameters
    ----------
    mm : float
        Distance in millimetres.
    pixel_spacing : float
        Size of one pixel in millimetres.

    Returns
    -------
    int
        Number of pixels (>= 1).
    """
    if pixel_spacing <= 0:
        raise ValueError(f"pixel_spacing must be positive, got {pixel_spacing}")
    return max(1, int(round(mm / pixel_spacing)))


def _extract_circular_roi(
    volume_slice: Any,
    center: Tuple[int, int],
    radius_px: int,
) -> Any:
    """Extract pixel values within a circular ROI from a 2-D slice.

    Parameters
    ----------
    volume_slice : np.ndarray
        2-D image array (Y, X).
    center : tuple[int, int]
        ROI centre as (row, col).
    radius_px : int
        Radius of the circular ROI in pixels.

    Returns
    -------
    np.ndarray
        1-D array of pixel values inside the circle.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    cy, cx = center
    rows, cols = volume_slice.shape
    yy, xx = np.ogrid[:rows, :cols]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius_px ** 2
    return volume_slice[mask]


def _find_phantom_center(volume_slice: Any) -> Tuple[int, int]:
    """Auto-detect the circular phantom centre using thresholding and centroid.

    Algorithm:
    1. Threshold the image to separate phantom from background air.
    2. Compute the centroid of the binary mask.

    Parameters
    ----------
    volume_slice : np.ndarray
        2-D image array (Y, X) in Hounsfield Units.

    Returns
    -------
    tuple[int, int]
        Detected phantom centre as (row, col).
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    # Threshold: phantom material is > -500 HU, background air is ~ -1000 HU
    threshold = -500.0
    binary = (volume_slice > threshold).astype(np.float64)

    if binary.sum() == 0:
        # Fallback: geometric centre of the image
        logger.warning(
            "Phantom centre detection failed (no pixels above threshold); "
            "using image centre."
        )
        return (volume_slice.shape[0] // 2, volume_slice.shape[1] // 2)

    rows, cols = volume_slice.shape
    yy, xx = np.mgrid[:rows, :cols]
    total = binary.sum()
    cy = int(round((yy * binary).sum() / total))
    cx = int(round((xx * binary).sum() / total))
    return (cy, cx)


def _find_phantom_rotation(volume_slice: Any) -> float:
    """Detect phantom rotation angle from insert positions.

    Uses the highest-HU insert (bone equivalent, ~955 HU) as the reference
    direction and computes the angular offset from the expected 90-degree
    position.

    Parameters
    ----------
    volume_slice : np.ndarray
        2-D image array (Y, X) in Hounsfield Units.

    Returns
    -------
    float
        Estimated rotation angle in degrees.  0.0 means no detectable rotation.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    center = _find_phantom_center(volume_slice)
    cy, cx = center
    rows, cols = volume_slice.shape

    # Search for the bone-equivalent insert (highest positive HU region
    # in the peripheral ring, expected ~955 HU).
    # Sample angular bins around the centre at the expected offset.
    offset_px = min(rows, cols) // 4  # approximate insert radial offset
    best_angle = 0.0
    best_mean = -1e9
    angles = np.arange(0, 360, 2.0)

    for angle in angles:
        rad = np.deg2rad(angle)
        sample_y = int(round(cy + offset_px * np.sin(rad)))
        sample_x = int(round(cx + offset_px * np.cos(rad)))
        # Ensure within bounds
        if 0 <= sample_y < rows and 0 <= sample_x < cols:
            # Small local average
            r = 3
            y_lo = max(0, sample_y - r)
            y_hi = min(rows, sample_y + r + 1)
            x_lo = max(0, sample_x - r)
            x_hi = min(cols, sample_x + r + 1)
            local_mean = float(volume_slice[y_lo:y_hi, x_lo:x_hi].mean())
            if local_mean > best_mean:
                best_mean = local_mean
                best_angle = float(angle)

    # Expected position of bone insert is at CW-from-12-o'clock 90° = 3 o'clock.
    # In the math convention used above (y = cy + offset*sin(a), x = cx + offset*cos(a)
    # with y-down image coordinates), 3 o'clock corresponds to 0°.
    rotation = best_angle - 0.0
    # Normalise to [-180, 180]
    if rotation > 180.0:
        rotation -= 360.0
    elif rotation < -180.0:
        rotation += 360.0

    return rotation


def _compute_cnr(signal_roi: Any, background_roi: Any) -> float:
    """Compute contrast-to-noise ratio between signal and background ROIs.

    Formula::

        CNR = |mean(signal) - mean(background)| / std(background)

    Parameters
    ----------
    signal_roi : np.ndarray
        Pixel values in the signal (target) ROI.
    background_roi : np.ndarray
        Pixel values in the background ROI.

    Returns
    -------
    float
        The CNR value.  Returns 0.0 if background std is zero.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    bg_std = float(np.std(background_roi, ddof=1)) if len(background_roi) > 1 else float(np.std(background_roi))
    if bg_std < 1e-12:
        return 0.0
    return float(abs(np.mean(signal_roi) - np.mean(background_roi)) / bg_std)


# ---------------------------------------------------------------------------
# Metric 1: CT Number -- Water
# ---------------------------------------------------------------------------

def compute_ct_number_water(
    volume: Any,
    center: Tuple[int, int],
    radius_mm: float,
    pixel_spacing: Tuple[float, float],
) -> MetricResult:
    """Compute mean Hounsfield Unit in the central water ROI.

    **ACR Criterion:** The mean CT number of water should be 0 +/- 5 HU
    for the central ROI of the ACR phantom Module 1.

    Parameters
    ----------
    volume : np.ndarray
        3-D volume (Z, Y, X) in HU.
    center : tuple[int, int]
        Centre of the water ROI as (row, col) in the central slice.
    radius_mm : float
        Radius of the circular ROI in mm.
    pixel_spacing : tuple[float, float]
        (row_spacing, col_spacing) in mm.

    Returns
    -------
    MetricResult
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    try:
        central_idx = volume.shape[0] // 2
        central_slice = volume[central_idx]
        radius_px = _mm_to_pixels(radius_mm, float(pixel_spacing[0]))
        roi_pixels = _extract_circular_roi(central_slice, center, radius_px)
        mean_hu = float(np.mean(roi_pixels))
        std_hu = float(np.std(roi_pixels))

        return MetricResult(
            name="CT Number - Water",
            value=round(mean_hu, 2),
            unit="HU",
            status="computed",
            derivation={
                "formula": "mean(ROI_pixels)",
                "std_hu": round(std_hu, 2),
                "num_pixels": int(roi_pixels.size),
                "acr_criterion": "0 +/- 5 HU",
                "slice_index": central_idx,
            },
            roi_info={
                "center_row": center[0],
                "center_col": center[1],
                "radius_mm": radius_mm,
                "radius_px": radius_px,
                "shape": "circle",
            },
        )
    except Exception as exc:
        logger.error("compute_ct_number_water failed: %s", exc)
        return MetricResult(
            name="CT Number - Water",
            value=float("nan"),
            unit="HU",
            status="error",
            derivation={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Metric 2: CT Number -- Insert
# ---------------------------------------------------------------------------

def compute_ct_number_insert(
    volume: Any,
    insert_name: str,
    center: Tuple[int, int],
    angle_deg: float,
    radius_mm: float,
    offset_mm: float,
    pixel_spacing: Tuple[float, float],
    expected_hu: float,
) -> MetricResult:
    """Compute mean HU in an insert ROI positioned at a given angle from centre.

    Used for bone, air, acrylic, and polyethylene inserts in Module 1.

    **ACR Criteria (typical):**
    - Bone: 850--1250 HU
    - Air: -1005 to -970 HU
    - Acrylic: 110--135 HU
    - Polyethylene: -107 to -84 HU

    Parameters
    ----------
    volume : np.ndarray
        3-D volume (Z, Y, X) in HU.
    insert_name : str
        Label for the insert (e.g. ``"bone"``).
    center : tuple[int, int]
        Phantom centre (row, col).
    angle_deg : float
        Angular position of the insert from 12-o'clock (degrees, CW).
    radius_mm : float
        ROI radius in mm.
    offset_mm : float
        Radial distance of the insert centre from phantom centre in mm.
    pixel_spacing : tuple[float, float]
        (row_spacing, col_spacing) in mm.
    expected_hu : float
        Nominal expected HU for the insert material.

    Returns
    -------
    MetricResult
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    try:
        central_idx = volume.shape[0] // 2
        central_slice = volume[central_idx]

        # Convert angle and offset to pixel coordinates
        angle_rad = np.deg2rad(angle_deg)
        offset_px_y = _mm_to_pixels(offset_mm, float(pixel_spacing[0]))
        offset_px_x = _mm_to_pixels(offset_mm, float(pixel_spacing[1]))
        radius_px = _mm_to_pixels(radius_mm, float(pixel_spacing[0]))

        # Insert position: angle is measured CW from 12-o'clock (negative Y)
        insert_row = int(round(center[0] - offset_px_y * np.cos(angle_rad)))
        insert_col = int(round(center[1] + offset_px_x * np.sin(angle_rad)))

        roi_pixels = _extract_circular_roi(
            central_slice, (insert_row, insert_col), radius_px
        )
        mean_hu = float(np.mean(roi_pixels))
        std_hu = float(np.std(roi_pixels))
        deviation = mean_hu - expected_hu

        return MetricResult(
            name=f"CT Number - {insert_name.capitalize()}",
            value=round(mean_hu, 2),
            unit="HU",
            status="computed",
            derivation={
                "formula": "mean(ROI_pixels)",
                "std_hu": round(std_hu, 2),
                "num_pixels": int(roi_pixels.size),
                "expected_hu": expected_hu,
                "deviation_hu": round(deviation, 2),
                "slice_index": central_idx,
            },
            roi_info={
                "insert_name": insert_name,
                "center_row": insert_row,
                "center_col": insert_col,
                "angle_deg": angle_deg,
                "offset_mm": offset_mm,
                "radius_mm": radius_mm,
                "radius_px": radius_px,
                "shape": "circle",
            },
        )
    except Exception as exc:
        logger.error("compute_ct_number_insert (%s) failed: %s", insert_name, exc)
        return MetricResult(
            name=f"CT Number - {insert_name.capitalize()}",
            value=float("nan"),
            unit="HU",
            status="error",
            derivation={"error": str(exc), "insert_name": insert_name},
        )


# ---------------------------------------------------------------------------
# Metric 3: Geometric Accuracy
# ---------------------------------------------------------------------------

def compute_geometric_accuracy(
    volume: Any,
    pixel_spacing: Tuple[float, float],
    expected_diameter_mm: float = 200.0,
) -> MetricResult:
    """Measure phantom diameter at 0 and 90 degrees, compare to expected.

    **ACR Criterion:** Measured diameter should be within +/- 2 mm of the
    expected value (200 mm for the ACR CT 464 phantom).

    Algorithm:
    1. Threshold the central slice to create a binary phantom mask.
    2. Measure the horizontal and vertical extents of the mask.
    3. Convert pixel extents to mm and compare with expected diameter.

    Parameters
    ----------
    volume : np.ndarray
        3-D volume (Z, Y, X) in HU.
    pixel_spacing : tuple[float, float]
        (row_spacing, col_spacing) in mm.
    expected_diameter_mm : float
        Expected phantom outer diameter in mm.

    Returns
    -------
    MetricResult
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    try:
        central_idx = volume.shape[0] // 2
        central_slice = volume[central_idx]

        # Threshold to isolate phantom from air
        threshold = -500.0
        binary = central_slice > threshold

        # Find bounding rows and columns
        row_any = np.any(binary, axis=1)
        col_any = np.any(binary, axis=0)
        row_indices = np.where(row_any)[0]
        col_indices = np.where(col_any)[0]

        if row_indices.size == 0 or col_indices.size == 0:
            raise ValueError("Could not detect phantom boundaries in central slice.")

        vertical_extent_px = int(row_indices[-1] - row_indices[0]) + 1
        horizontal_extent_px = int(col_indices[-1] - col_indices[0]) + 1

        vertical_mm = vertical_extent_px * float(pixel_spacing[0])
        horizontal_mm = horizontal_extent_px * float(pixel_spacing[1])

        # Report the maximum deviation from expected
        vert_error = vertical_mm - expected_diameter_mm
        horiz_error = horizontal_mm - expected_diameter_mm
        max_abs_error = max(abs(vert_error), abs(horiz_error))

        return MetricResult(
            name="Geometric Accuracy",
            value=round(max_abs_error, 2),
            unit="mm",
            status="computed",
            derivation={
                "formula": "max(|measured_diameter - expected_diameter|) at 0deg, 90deg",
                "vertical_diameter_mm": round(vertical_mm, 2),
                "horizontal_diameter_mm": round(horizontal_mm, 2),
                "vertical_error_mm": round(vert_error, 2),
                "horizontal_error_mm": round(horiz_error, 2),
                "expected_diameter_mm": expected_diameter_mm,
                "acr_criterion": "within +/- 2 mm",
                "slice_index": central_idx,
            },
            roi_info={
                "measurement": "bounding_box",
                "row_range": [int(row_indices[0]), int(row_indices[-1])],
                "col_range": [int(col_indices[0]), int(col_indices[-1])],
            },
        )
    except Exception as exc:
        logger.error("compute_geometric_accuracy failed: %s", exc)
        return MetricResult(
            name="Geometric Accuracy",
            value=float("nan"),
            unit="mm",
            status="error",
            derivation={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Metric 4: Slice Thickness
# ---------------------------------------------------------------------------

def compute_slice_thickness(
    volume: Any,
    slice_thickness_nominal: float,
    wire_ramp_profile: Any = None,
    pixel_spacing: float = 1.0,
) -> MetricResult:
    """Compute slice thickness from FWHM of wire-ramp response profile.

    **ACR Criterion:** Measured slice thickness should be within +/- 1.5 mm
    of the nominal value.

    For the ACR CT 464 phantom Module 1, two crossed wire ramps produce
    a trapezoidal profile.  The FWHM of this profile gives the effective
    slice thickness.

    Parameters
    ----------
    volume : np.ndarray
        3-D volume (Z, Y, X) in HU.
    slice_thickness_nominal : float
        Nominal (prescribed) slice thickness in mm.
    wire_ramp_profile : np.ndarray, optional
        Pre-extracted 1-D profile along the ramp.  If *None*, the function
        attempts to extract the profile from the central slice.

    Returns
    -------
    MetricResult
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    try:
        if wire_ramp_profile is not None:
            profile = np.asarray(wire_ramp_profile, dtype=np.float64)
        else:
            # Auto-extract: take a horizontal line through the centre of the
            # central slice and look for the ramp signal (peak region)
            central_idx = volume.shape[0] // 2
            central_slice = volume[central_idx]
            center_row = central_slice.shape[0] // 2
            profile = central_slice[center_row, :].astype(np.float64)

        # Compute FWHM
        profile_min = float(np.min(profile))
        profile_max = float(np.max(profile))
        half_max = (profile_max + profile_min) / 2.0

        above_half = profile >= half_max
        # Find first and last crossing of half-max
        transitions = np.diff(above_half.astype(np.int8))
        rise_indices = np.where(transitions == 1)[0]
        fall_indices = np.where(transitions == -1)[0]

        if rise_indices.size == 0 or fall_indices.size == 0:
            # Fallback: count pixels above half-max
            fwhm_px = float(np.sum(above_half))
        else:
            # Interpolate for sub-pixel accuracy on the first peak
            rise = int(rise_indices[0])
            fall = int(fall_indices[-1]) if fall_indices[-1] > rise else int(fall_indices[0])
            fwhm_px = float(fall - rise)
            if fwhm_px <= 0:
                fwhm_px = float(np.sum(above_half))

        # For the ACR phantom ramp, the FWHM in pixels needs to be
        # converted to mm using the ramp geometry.  The standard ACR
        # ramp has a 23-degree angle, so:
        #   slice_thickness = FWHM_mm * tan(23 deg)
        # But if the profile is along the Z axis, FWHM_mm is
        # directly the slice thickness.
        #
        # Here we assume the profile is sampled along the in-plane
        # direction with pixel_spacing, and the ramp angle converts it:
        ramp_angle_deg = 23.0
        ramp_angle_rad = np.deg2rad(ramp_angle_deg)
        # In-plane FWHM in mm (assuming pixel_spacing ~ 0.5 mm typical)
        # For a pre-extracted profile, pixel spacing is already baked in
        fwhm_mm = fwhm_px * pixel_spacing
        measured_thickness = fwhm_mm * np.tan(ramp_angle_rad)

        error_mm = measured_thickness - slice_thickness_nominal

        return MetricResult(
            name="Slice Thickness",
            value=round(measured_thickness, 2),
            unit="mm",
            status="computed",
            derivation={
                "formula": "FWHM_mm * tan(ramp_angle)",
                "fwhm_px": round(fwhm_px, 2),
                "pixel_spacing": pixel_spacing,
                "fwhm_mm": round(fwhm_mm, 2),
                "ramp_angle_deg": ramp_angle_deg,
                "nominal_mm": slice_thickness_nominal,
                "error_mm": round(error_mm, 2),
                "acr_criterion": "within +/- 1.5 mm of nominal",
            },
            roi_info={
                "profile_source": (
                    "provided" if wire_ramp_profile is not None else "auto_central_row"
                ),
                "profile_length": int(profile.size),
            },
        )
    except Exception as exc:
        logger.error("compute_slice_thickness failed: %s", exc)
        return MetricResult(
            name="Slice Thickness",
            value=float("nan"),
            unit="mm",
            status="error",
            derivation={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Metric 5: Uniformity
# ---------------------------------------------------------------------------

def compute_uniformity(
    volume: Any,
    center: Tuple[int, int],
    radius_mm: float,
    peripheral_offset_mm: float,
    pixel_spacing: Tuple[float, float],
) -> MetricResult:
    """Compute CT number uniformity from centre and 4 peripheral ROIs.

    **ACR Criterion:** The maximum difference between any peripheral ROI
    mean and the centre ROI mean should be <= 5 HU.

    Peripheral ROIs are placed at 12, 3, 6, and 9 o'clock positions
    at the specified offset from the phantom centre.

    Parameters
    ----------
    volume : np.ndarray
        3-D volume (Z, Y, X) in HU.
    center : tuple[int, int]
        Phantom centre (row, col).
    radius_mm : float
        Radius of each ROI in mm.
    peripheral_offset_mm : float
        Distance from phantom centre to peripheral ROI centres in mm.
    pixel_spacing : tuple[float, float]
        (row_spacing, col_spacing) in mm.

    Returns
    -------
    MetricResult
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    try:
        central_idx = volume.shape[0] // 2
        central_slice = volume[central_idx]

        radius_px = _mm_to_pixels(radius_mm, float(pixel_spacing[0]))
        offset_px_y = _mm_to_pixels(peripheral_offset_mm, float(pixel_spacing[0]))
        offset_px_x = _mm_to_pixels(peripheral_offset_mm, float(pixel_spacing[1]))

        # Centre ROI
        center_roi = _extract_circular_roi(central_slice, center, radius_px)
        center_mean = float(np.mean(center_roi))

        # Peripheral ROI positions (12, 3, 6, 9 o'clock)
        peripheral_positions = {
            "12_oclock": (center[0] - offset_px_y, center[1]),
            "3_oclock": (center[0], center[1] + offset_px_x),
            "6_oclock": (center[0] + offset_px_y, center[1]),
            "9_oclock": (center[0], center[1] - offset_px_x),
        }

        peripheral_means: Dict[str, float] = {}
        max_diff = 0.0
        for pos_name, (py, px) in peripheral_positions.items():
            roi = _extract_circular_roi(central_slice, (int(py), int(px)), radius_px)
            p_mean = float(np.mean(roi))
            peripheral_means[pos_name] = round(p_mean, 2)
            diff = abs(p_mean - center_mean)
            if diff > max_diff:
                max_diff = diff

        return MetricResult(
            name="Uniformity",
            value=round(max_diff, 2),
            unit="HU",
            status="computed",
            derivation={
                "formula": "max(|peripheral_mean - center_mean|) over 4 positions",
                "center_mean_hu": round(center_mean, 2),
                "peripheral_means_hu": peripheral_means,
                "acr_criterion": "<= 5 HU",
                "slice_index": central_idx,
            },
            roi_info={
                "center": {"row": center[0], "col": center[1]},
                "radius_mm": radius_mm,
                "radius_px": radius_px,
                "peripheral_offset_mm": peripheral_offset_mm,
                "peripheral_positions": {
                    k: {"row": int(v[0]), "col": int(v[1])}
                    for k, v in peripheral_positions.items()
                },
            },
        )
    except Exception as exc:
        logger.error("compute_uniformity failed: %s", exc)
        return MetricResult(
            name="Uniformity",
            value=float("nan"),
            unit="HU",
            status="error",
            derivation={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Metric 6: Noise (Standard Deviation)
# ---------------------------------------------------------------------------

def compute_noise_std(
    volume: Any,
    center: Tuple[int, int],
    radius_mm: float,
    pixel_spacing: Tuple[float, float],
) -> MetricResult:
    """Compute noise as the standard deviation in a uniform water ROI.

    The noise value is compared to the scanner's established baseline.
    A significant increase may indicate tube degradation or calibration drift.

    Parameters
    ----------
    volume : np.ndarray
        3-D volume (Z, Y, X) in HU.
    center : tuple[int, int]
        Centre of the uniform water ROI (row, col).
    radius_mm : float
        ROI radius in mm.
    pixel_spacing : tuple[float, float]
        (row_spacing, col_spacing) in mm.

    Returns
    -------
    MetricResult
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    try:
        central_idx = volume.shape[0] // 2
        central_slice = volume[central_idx]
        radius_px = _mm_to_pixels(radius_mm, float(pixel_spacing[0]))
        roi_pixels = _extract_circular_roi(central_slice, center, radius_px)
        noise_std = float(np.std(roi_pixels, ddof=1)) if roi_pixels.size > 1 else float(np.std(roi_pixels))
        mean_hu = float(np.mean(roi_pixels))

        return MetricResult(
            name="Noise (Std Dev)",
            value=round(noise_std, 2),
            unit="HU",
            status="computed",
            derivation={
                "formula": "std(ROI_pixels)",
                "mean_hu": round(mean_hu, 2),
                "num_pixels": int(roi_pixels.size),
                "note": "Compare to scanner baseline",
                "slice_index": central_idx,
            },
            roi_info={
                "center_row": center[0],
                "center_col": center[1],
                "radius_mm": radius_mm,
                "radius_px": radius_px,
                "shape": "circle",
            },
        )
    except Exception as exc:
        logger.error("compute_noise_std failed: %s", exc)
        return MetricResult(
            name="Noise (Std Dev)",
            value=float("nan"),
            unit="HU",
            status="error",
            derivation={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Metric 7: Low-Contrast Detectability
# ---------------------------------------------------------------------------

def compute_low_contrast_detectability(
    volume: Any,
    target_sizes: List[float],
    contrast_hu: float,
    pixel_spacing: Tuple[float, float],
) -> MetricResult:
    """Count visible low-contrast targets using a CNR-based detection criterion.

    **ACR Criterion:** At least 4 targets should be visible (CNR >= threshold)
    in Module 3 of the ACR CT 464 phantom.

    For each target size, a circular ROI of that diameter is placed at the
    expected target location, and the CNR is computed against the adjacent
    background.  A target is considered "detected" if CNR >= 1.0 (Rose
    criterion simplified).

    Parameters
    ----------
    volume : np.ndarray
        3-D volume (Z, Y, X) in HU.
    target_sizes : list[float]
        List of target diameters in mm (e.g. [2, 3, 4, 5, 6]).
    contrast_hu : float
        Nominal contrast of the targets in HU.
    pixel_spacing : tuple[float, float]
        (row_spacing, col_spacing) in mm.

    Returns
    -------
    MetricResult
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    try:
        central_idx = volume.shape[0] // 2
        central_slice = volume[central_idx]
        center = _find_phantom_center(central_slice)

        detected_count = 0
        target_details: List[Dict[str, Any]] = []

        # Background ROI: large region at phantom centre
        bg_radius_px = _mm_to_pixels(20.0, float(pixel_spacing[0]))
        background_roi = _extract_circular_roi(central_slice, center, bg_radius_px)

        for target_size_mm in sorted(target_sizes, reverse=True):
            target_radius_mm = target_size_mm / 2.0
            target_radius_px = _mm_to_pixels(target_radius_mm, float(pixel_spacing[0]))

            # Place target ROI at an offset from centre
            # (simplified: use concentric ring at 40mm offset)
            offset_px = _mm_to_pixels(40.0, float(pixel_spacing[0]))
            target_center = (center[0] - offset_px, center[1])

            if (
                target_center[0] - target_radius_px < 0
                or target_center[0] + target_radius_px >= central_slice.shape[0]
            ):
                target_details.append({
                    "size_mm": target_size_mm,
                    "detected": False,
                    "reason": "target ROI out of bounds",
                })
                continue

            target_roi = _extract_circular_roi(
                central_slice, target_center, target_radius_px
            )

            if target_roi.size == 0:
                target_details.append({
                    "size_mm": target_size_mm,
                    "detected": False,
                    "reason": "empty ROI",
                })
                continue

            cnr = _compute_cnr(target_roi, background_roi)
            detected = cnr >= 1.0  # Rose criterion (simplified)

            if detected:
                detected_count += 1

            target_details.append({
                "size_mm": target_size_mm,
                "cnr": round(cnr, 3),
                "detected": detected,
                "target_mean_hu": round(float(np.mean(target_roi)), 2),
            })

        return MetricResult(
            name="Low-Contrast Detectability",
            value=float(detected_count),
            unit="targets",
            status="computed",
            derivation={
                "formula": "count of targets with CNR >= 1.0",
                "cnr_threshold": 1.0,
                "nominal_contrast_hu": contrast_hu,
                "target_details": target_details,
                "acr_criterion": ">= 4 targets visible",
                "slice_index": central_idx,
            },
            roi_info={
                "phantom_center": {"row": center[0], "col": center[1]},
                "background_radius_mm": 20.0,
            },
        )
    except Exception as exc:
        logger.error("compute_low_contrast_detectability failed: %s", exc)
        return MetricResult(
            name="Low-Contrast Detectability",
            value=float("nan"),
            unit="targets",
            status="error",
            derivation={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Metric 8: Artifact Evaluation
# ---------------------------------------------------------------------------

def compute_artifact_evaluation(
    volume: Any,
    center: Tuple[int, int],
    radius_mm: float,
    pixel_spacing: Tuple[float, float],
) -> MetricResult:
    """Detect ring and streak artifacts in a uniform region of the phantom.

    Scoring:
    - 0: No artifacts detected (peak-to-valley < 3 HU)
    - 1: Minor artifacts (3--5 HU)
    - 2: Moderate artifacts (5--8 HU)
    - 3: Severe artifacts (> 8 HU)

    Algorithm:
    1. Extract a large circular ROI in the uniform water region.
    2. Compute the radial average profile from the centre outward.
    3. Measure peak-to-valley variation in the radial profile.

    Parameters
    ----------
    volume : np.ndarray
        3-D volume (Z, Y, X) in HU.
    center : tuple[int, int]
        Phantom centre (row, col).
    radius_mm : float
        Radius of the evaluation region in mm.
    pixel_spacing : tuple[float, float]
        (row_spacing, col_spacing) in mm.

    Returns
    -------
    MetricResult
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    try:
        central_idx = volume.shape[0] // 2
        central_slice = volume[central_idx]
        radius_px = _mm_to_pixels(radius_mm, float(pixel_spacing[0]))

        cy, cx = center
        rows, cols = central_slice.shape

        # Build radial profile
        max_r = min(radius_px, cy, rows - cy - 1, cx, cols - cx - 1)
        if max_r < 5:
            raise ValueError("Evaluation region too small for artifact analysis.")

        radial_bins: Dict[int, List[float]] = {}
        for dy in range(-max_r, max_r + 1):
            for dx in range(-max_r, max_r + 1):
                r = int(round((dy**2 + dx**2) ** 0.5))
                if r > max_r:
                    continue
                py = cy + dy
                px_coord = cx + dx
                if 0 <= py < rows and 0 <= px_coord < cols:
                    radial_bins.setdefault(r, []).append(
                        float(central_slice[py, px_coord])
                    )

        # Compute mean at each radial distance
        radii = sorted(radial_bins.keys())
        radial_means = np.array([np.mean(radial_bins[r]) for r in radii])

        # Peak-to-valley in the radial profile
        peak_to_valley = float(np.max(radial_means) - np.min(radial_means))

        # Score
        if peak_to_valley < 3.0:
            score = 0
            grade = "none"
        elif peak_to_valley < 8.0:
            score = 1 if peak_to_valley < 5.0 else 2
            grade = "minor" if score == 1 else "moderate"
        else:
            score = 3
            grade = "severe"

        return MetricResult(
            name="Artifact Evaluation",
            value=float(score),
            unit="score (0-3)",
            status="computed",
            derivation={
                "formula": "score from peak-to-valley of radial profile",
                "peak_to_valley_hu": round(peak_to_valley, 2),
                "grade": grade,
                "thresholds": {"none": "<3 HU", "minor": "3-5 HU", "moderate": "5-8 HU", "severe": ">8 HU"},
                "num_radial_bins": len(radii),
                "max_radius_px": max_r,
                "slice_index": central_idx,
            },
            roi_info={
                "center_row": center[0],
                "center_col": center[1],
                "radius_mm": radius_mm,
                "radius_px": radius_px,
                "shape": "circle",
            },
        )
    except Exception as exc:
        logger.error("compute_artifact_evaluation failed: %s", exc)
        return MetricResult(
            name="Artifact Evaluation",
            value=float("nan"),
            unit="score (0-3)",
            status="error",
            derivation={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Metric 9: Spatial Resolution
# ---------------------------------------------------------------------------

def compute_spatial_resolution(
    volume: Any,
    pixel_spacing: Tuple[float, float],
    line_pair_groups: Optional[List[int]] = None,
) -> MetricResult:
    """Estimate spatial resolution via MTF analysis of the reconstructed image.

    **ACR Criterion:** At least 5 lp/cm should be resolvable in Module 4
    of the ACR CT 464 phantom.

    This implementation computes the Modulation Transfer Function (MTF) from
    the edge-spread function (ESF) of the phantom boundary, then determines
    the highest spatial frequency at which MTF >= 10%.

    Parameters
    ----------
    volume : np.ndarray
        3-D volume (Z, Y, X) in HU.
    pixel_spacing : tuple[float, float]
        (row_spacing, col_spacing) in mm.
    line_pair_groups : list[int], optional
        Available line-pair groups per cm (e.g. [4, 5, 6, 7, 8]).
        Default is [4, 5, 6, 7, 8].

    Returns
    -------
    MetricResult
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    if line_pair_groups is None:
        line_pair_groups = [4, 5, 6, 7, 8]

    try:
        central_idx = volume.shape[0] // 2
        central_slice = volume[central_idx]
        center = _find_phantom_center(central_slice)
        cy, cx = center

        # Extract a horizontal edge-spread function through the phantom boundary
        row_profile = central_slice[cy, :].astype(np.float64)

        # Differentiate to get the line-spread function (LSF)
        lsf = np.diff(row_profile)

        # Zero-pad and compute FFT for the MTF
        n = len(lsf)
        padded_length = max(512, 2 ** int(np.ceil(np.log2(n))))
        lsf_padded = np.zeros(padded_length)
        lsf_padded[:n] = lsf

        mtf_raw = np.abs(np.fft.rfft(lsf_padded))
        if mtf_raw[0] > 0:
            mtf = mtf_raw / mtf_raw[0]  # Normalise to 1 at DC
        else:
            mtf = mtf_raw

        # Frequency axis in lp/cm
        # Nyquist frequency = 1 / (2 * pixel_spacing_mm) = 1 / (2 * ps) [lp/mm]
        # Convert to lp/cm: multiply by 10
        freq_axis_lp_per_mm = np.linspace(
            0, 1.0 / (2.0 * float(pixel_spacing[1])), len(mtf)
        )
        freq_axis_lp_per_cm = freq_axis_lp_per_mm * 10.0

        # Find the highest frequency where MTF >= 10%
        mtf_threshold = 0.10
        above_threshold = freq_axis_lp_per_cm[mtf >= mtf_threshold]
        max_resolvable = float(above_threshold[-1]) if above_threshold.size > 0 else 0.0

        # Map to the highest standard line-pair group that is resolvable
        resolvable_groups = [g for g in line_pair_groups if g <= max_resolvable]
        highest_group = max(resolvable_groups) if resolvable_groups else 0

        return MetricResult(
            name="Spatial Resolution",
            value=round(max_resolvable, 1),
            unit="lp/cm",
            status="computed",
            derivation={
                "formula": "MTF-based: highest freq where MTF >= 10%",
                "mtf_threshold": mtf_threshold,
                "max_resolvable_lp_cm": round(max_resolvable, 2),
                "highest_acr_group": highest_group,
                "line_pair_groups_tested": line_pair_groups,
                "acr_criterion": ">= 5 lp/cm",
                "slice_index": central_idx,
            },
            roi_info={
                "method": "edge_spread_function",
                "profile_row": cy,
                "profile_length": n,
                "fft_length": padded_length,
            },
        )
    except Exception as exc:
        logger.error("compute_spatial_resolution failed: %s", exc)
        return MetricResult(
            name="Spatial Resolution",
            value=float("nan"),
            unit="lp/cm",
            status="error",
            derivation={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_all_metrics(
    scan_bundle: Any,
    casepack: Any,
) -> QAMetricsReport:
    """Compute all CasePack-specified CT QA metrics for a scan.

    Iterates over ``casepack.metric_set`` (or ``casepack["casepack"]["metric_set"]``
    if a raw dict), calls the appropriate metric function, and collects
    results.  Individual metric failures are caught so one bad metric does
    not crash the entire report.

    Parameters
    ----------
    scan_bundle : CTScanBundle
        The ingested scan data (from :class:`DICOMIngester`).
    casepack : Any
        Parsed CasePack configuration (dict or object).

    Returns
    -------
    QAMetricsReport
        Complete metrics report with timing information.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required.") from exc

    t0 = time.monotonic()

    # Parse casepack configuration (supports both dict and CasePackConfig)
    if isinstance(casepack, dict):
        cp = casepack.get("casepack", casepack)
    else:
        cp = casepack

    if isinstance(cp, dict):
        casepack_id = cp.get("id", "unknown")
        metric_set = cp.get("metric_set", [])
        roi_defs = cp.get("roi_definitions", {})
    else:
        # CasePackConfig or other Pydantic model
        casepack_id = getattr(cp, "id", "unknown")
        metric_set = getattr(cp, "metric_set", [])
        roi_defs = getattr(cp, "roi_definitions", {})

    insert_rois: dict = roi_defs.get("insert_rois", {}) if isinstance(roi_defs, dict) else {}

    # Extract common parameters from scan_bundle
    volume = scan_bundle.slices
    pixel_spacing = scan_bundle.pixel_spacing_mm
    scanner_id = scan_bundle.scanner_id
    study_date = scan_bundle.study_date

    # Auto-detect phantom centre from central slice
    central_idx = volume.shape[0] // 2
    center = _find_phantom_center(volume[central_idx])
    rotation_offset = _find_phantom_rotation(volume[central_idx])

    # ROI configuration defaults (from casepack or fallback)
    water_roi_cfg = roi_defs.get("water_roi", {})
    water_radius_mm: float = float(water_roi_cfg.get("radius_mm", 20.0))

    peripheral_cfg = roi_defs.get("peripheral_rois", {})
    peripheral_radius_mm: float = float(peripheral_cfg.get("radius_mm", 10.0))
    peripheral_offset_mm: float = float(peripheral_cfg.get("offset_from_center_mm", 60.0))

    # Spatial resolution config
    spatial_cfg = roi_defs.get("spatial_resolution_patterns", {})
    lp_groups: list[int] = spatial_cfg.get("groups", [4, 5, 6, 7, 8])

    # Low-contrast config
    lc_cfg = roi_defs.get("low_contrast_targets", {})
    target_sizes: list[float] = lc_cfg.get("target_sizes_mm", [2, 3, 4, 5, 6])
    contrast_hu: float = float(lc_cfg.get("nominal_contrast_hu", 6.0))

    # Slice thickness config
    ramp_cfg = roi_defs.get("slice_thickness_ramp", {})
    nominal_slice_thickness = scan_bundle.slice_thickness_mm

    # Dispatch table
    metrics: Dict[str, MetricResult] = {}

    for metric_key in metric_set:
        try:
            if metric_key == "ct_number_water":
                metrics[metric_key] = compute_ct_number_water(
                    volume, center, water_radius_mm, pixel_spacing
                )

            elif metric_key.startswith("ct_number_") and metric_key != "ct_number_water":
                # Insert-based CT number metric
                insert_name = metric_key.replace("ct_number_", "")
                insert_cfg = insert_rois.get(insert_name, {})

                if not insert_cfg:
                    metrics[metric_key] = MetricResult(
                        name=f"CT Number - {insert_name.capitalize()}",
                        value=float("nan"),
                        unit="HU",
                        status="skipped",
                        derivation={
                            "reason": f"No insert ROI config for '{insert_name}'"
                        },
                    )
                    continue

                metrics[metric_key] = compute_ct_number_insert(
                    volume=volume,
                    insert_name=insert_name,
                    center=center,
                    angle_deg=float(insert_cfg.get("offset_angle_deg", 0)) + rotation_offset,
                    radius_mm=float(insert_cfg.get("roi_radius_mm", 5.0)),
                    offset_mm=float(insert_cfg.get("offset_radius_mm", 50)),
                    pixel_spacing=pixel_spacing,
                    expected_hu=float(insert_cfg.get("expected_hu", 0)),
                )

            elif metric_key == "geometric_accuracy":
                metrics[metric_key] = compute_geometric_accuracy(
                    volume, pixel_spacing
                )

            elif metric_key == "slice_thickness":
                metrics[metric_key] = compute_slice_thickness(
                    volume, nominal_slice_thickness,
                    pixel_spacing=float(pixel_spacing[1]),
                )

            elif metric_key == "uniformity":
                metrics[metric_key] = compute_uniformity(
                    volume, center, peripheral_radius_mm,
                    peripheral_offset_mm, pixel_spacing
                )

            elif metric_key == "noise_std":
                metrics[metric_key] = compute_noise_std(
                    volume, center, water_radius_mm, pixel_spacing
                )

            elif metric_key == "low_contrast_detectability":
                metrics[metric_key] = compute_low_contrast_detectability(
                    volume, target_sizes, contrast_hu, pixel_spacing
                )

            elif metric_key == "artifact_evaluation":
                metrics[metric_key] = compute_artifact_evaluation(
                    volume, center, water_radius_mm, pixel_spacing
                )

            elif metric_key == "spatial_resolution":
                metrics[metric_key] = compute_spatial_resolution(
                    volume, pixel_spacing, lp_groups
                )

            else:
                logger.warning("Unknown metric key '%s'; skipping.", metric_key)
                metrics[metric_key] = MetricResult(
                    name=metric_key,
                    value=float("nan"),
                    unit="",
                    status="skipped",
                    derivation={"reason": f"Unknown metric key: {metric_key}"},
                )

        except Exception as exc:
            logger.error("Metric '%s' raised: %s", metric_key, exc)
            metrics[metric_key] = MetricResult(
                name=metric_key,
                value=float("nan"),
                unit="",
                status="error",
                derivation={"error": str(exc)},
            )

    elapsed = time.monotonic() - t0

    return QAMetricsReport(
        scanner_id=scanner_id,
        date=study_date,
        casepack_id=casepack_id,
        metrics=metrics,
        computation_time_sec=round(elapsed, 4),
    )
