"""Artifact signature feature extraction for CT QC diagnosis.

Computes quantitative features from image data that indicate specific
types of scanner mismatch or malfunction. All features are unit-testable
with synthetic artifact injection.

Each feature function returns a single scalar that quantifies one aspect
of image quality degradation:

* **Ring index** -- azimuthal variance in polar coordinates (detector gain
  drift, centre-of-rotation offset).
* **Cupping index** -- radial HU profile shape (scatter, beam hardening).
* **Streak index** -- directional gradient anisotropy (dead channels, metal).
* **HU drift index** -- RMS deviation of insert HU from baseline values.
* **Noise ratio** -- current noise vs. baseline noise.
* **Geometric distortion index** -- RMS of relative distance errors.

Usage::

    from pwm_core.clinical.ct.diagnosis_features import compute_all_features

    features = compute_all_features(
        volume=slice_data,
        center=(256, 256),
        radius_px=200,
        current_metrics=today_metrics,
        baseline_metrics=commissioning_metrics,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Optional dependency: numpy (should always be available in a scientific
# Python environment, but we guard the import for robustness).
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class ArtifactFeatures(BaseModel):
    """Complete set of artifact signature features for a CT slice.

    All values are non-negative scalars.  Higher values indicate
    greater severity for the corresponding artifact type.

    Attributes
    ----------
    ring_index : float
        Azimuthal variance ratio indicating ring artifacts.
    cupping_index : float
        Centre-periphery HU difference (positive = cupping).
    streak_index : float
        Directional gradient anisotropy indicating streaks.
    hu_drift_index : float
        RMS deviation of insert HU values from baseline.
    noise_ratio : float
        Current-to-baseline noise standard deviation ratio.
    geometric_distortion_index : float
        RMS of relative geometric distance errors.
    """

    ring_index: float
    cupping_index: float
    streak_index: float
    hu_drift_index: float
    noise_ratio: float
    geometric_distortion_index: float


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def compute_ring_index(
    volume_slice: np.ndarray,
    center: tuple[int, int],
    inner_radius_px: int,
    outer_radius_px: int,
) -> float:
    """Quantify ring artifacts via azimuthal variance in polar coordinates.

    The image is re-sampled into polar coordinates centred on the phantom.
    For each radial bin the azimuthal (angular) variance is computed.  The
    ring index is the mean azimuthal variance normalised by the overall
    image variance within the annular region.

    Parameters
    ----------
    volume_slice : np.ndarray
        2-D image slice (H x W), typically in HU.
    center : tuple[int, int]
        Phantom centre coordinates ``(row, col)``.
    inner_radius_px : int
        Inner radius of the annular analysis region (pixels).
    outer_radius_px : int
        Outer radius of the annular analysis region (pixels).

    Returns
    -------
    float
        Ring index >= 0.  Values near 0 indicate no ring artifacts;
        values >> 1 indicate strong azimuthal structure (rings).
    """
    _require_numpy()
    volume_slice = volume_slice.astype(np.float64)
    cy, cx = center

    # Build coordinate grids
    rows, cols = volume_slice.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Annular mask
    mask = (rr >= inner_radius_px) & (rr < outer_radius_px)
    if mask.sum() == 0:
        logger.warning("Empty annular region; returning ring_index = 0.0.")
        return 0.0

    # Overall variance in the annular region
    annular_values = volume_slice[mask]
    overall_var = float(np.var(annular_values))
    if overall_var < 1e-12:
        return 0.0

    # Compute azimuthal variance at each integer radius
    num_angular_bins = 360
    theta = np.arctan2(yy - cy, xx - cx)  # range [-pi, pi]

    azimuthal_variances: list[float] = []
    for r in range(inner_radius_px, outer_radius_px):
        ring_mask = (rr >= r) & (rr < r + 1)
        ring_pixels = volume_slice[ring_mask]
        if len(ring_pixels) < num_angular_bins // 4:
            continue
        azimuthal_variances.append(float(np.var(ring_pixels)))

    if not azimuthal_variances:
        return 0.0

    mean_azimuthal_var = float(np.mean(azimuthal_variances))
    ring_index = mean_azimuthal_var / overall_var

    return ring_index


def compute_cupping_index(
    volume_slice: np.ndarray,
    center: tuple[int, int],
    radius_px: int,
) -> float:
    """Quantify cupping/capping from radial HU profile shape.

    Cupping (positive index) indicates the centre is darker than the
    periphery -- typical of scatter or beam-hardening effects.  Capping
    (negative index) indicates the reverse.

    Parameters
    ----------
    volume_slice : np.ndarray
        2-D image slice (H x W), typically in HU.
    center : tuple[int, int]
        Phantom centre coordinates ``(row, col)``.
    radius_px : int
        Phantom radius in pixels.

    Returns
    -------
    float
        Cupping index (HU).  Positive = cupping, negative = capping.
    """
    _require_numpy()
    volume_slice = volume_slice.astype(np.float64)
    cy, cx = center

    rows, cols = volume_slice.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Central region: inner 30 % of radius
    inner_limit = 0.30 * radius_px
    center_mask = rr <= inner_limit
    if center_mask.sum() == 0:
        return 0.0
    center_mean = float(np.mean(volume_slice[center_mask]))

    # Peripheral ring: 70-90 % of radius
    periph_inner = 0.70 * radius_px
    periph_outer = 0.90 * radius_px
    periph_mask = (rr >= periph_inner) & (rr <= periph_outer)
    if periph_mask.sum() == 0:
        return 0.0
    periph_mean = float(np.mean(volume_slice[periph_mask]))

    cupping_index = center_mean - periph_mean
    return cupping_index


def compute_streak_index(
    volume_slice: np.ndarray,
    center: tuple[int, int],
    radius_px: int,
) -> float:
    """Quantify streak artifacts via directional gradient anisotropy.

    A Sobel filter is applied in both X and Y directions.  The directional
    gradient energy is computed as the ratio of the maximum directional
    energy to the mean directional energy.  Isotropic noise yields a ratio
    near 1; strong streaks produce a high ratio.

    Parameters
    ----------
    volume_slice : np.ndarray
        2-D image slice (H x W), typically in HU.
    center : tuple[int, int]
        Phantom centre coordinates ``(row, col)``.
    radius_px : int
        Phantom radius in pixels (defines the analysis region).

    Returns
    -------
    float
        Streak index >= 1.  Values near 1.0 indicate no directional
        preference; values >> 1 indicate strong streak artefacts.
    """
    _require_numpy()
    volume_slice = volume_slice.astype(np.float64)
    cy, cx = center

    # Build circular mask within the phantom
    rows, cols = volume_slice.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = rr <= radius_px

    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = _convolve2d(volume_slice, sobel_x)
    gy = _convolve2d(volume_slice, sobel_y)

    # Compute directional energies within the phantom mask
    energy_x = float(np.mean(gx[mask] ** 2)) if mask.sum() > 0 else 0.0
    energy_y = float(np.mean(gy[mask] ** 2)) if mask.sum() > 0 else 0.0

    # Compute across multiple angles (0, 45, 90, 135 degrees)
    # For 45 and 135 degrees, combine Sobel outputs
    cos45 = math.cos(math.radians(45))
    sin45 = math.sin(math.radians(45))
    energy_45 = float(np.mean((gx[mask] * cos45 + gy[mask] * sin45) ** 2)) if mask.sum() > 0 else 0.0
    energy_135 = float(np.mean((gx[mask] * (-cos45) + gy[mask] * sin45) ** 2)) if mask.sum() > 0 else 0.0

    energies = [energy_x, energy_y, energy_45, energy_135]
    mean_energy = sum(energies) / len(energies)

    if mean_energy < 1e-12:
        return 1.0

    max_energy = max(energies)
    streak_index = max_energy / mean_energy

    return streak_index


def compute_hu_drift_index(
    current_inserts: dict[str, float],
    baseline_inserts: dict[str, float],
) -> float:
    """Quantify HU calibration drift across material inserts.

    Computes the RMS deviation of current insert HU values from baseline
    values, normalised by the expected total HU range across all inserts
    in the baseline.

    Parameters
    ----------
    current_inserts : dict[str, float]
        Current measured HU values keyed by insert name.
    baseline_inserts : dict[str, float]
        Baseline (commissioning) HU values keyed by insert name.

    Returns
    -------
    float
        HU drift index >= 0.  Values near 0 indicate stable calibration.
    """
    common_keys = set(current_inserts.keys()) & set(baseline_inserts.keys())
    if not common_keys:
        logger.warning("No common inserts between current and baseline; returning 0.0.")
        return 0.0

    baseline_vals = [baseline_inserts[k] for k in common_keys]
    expected_range = max(baseline_vals) - min(baseline_vals)
    if expected_range < 1e-12:
        expected_range = 1.0  # Avoid division by zero

    squared_diffs: list[float] = []
    for key in common_keys:
        diff = current_inserts[key] - baseline_inserts[key]
        squared_diffs.append(diff ** 2)

    rms = math.sqrt(sum(squared_diffs) / len(squared_diffs))
    hu_drift_index = rms / expected_range

    return hu_drift_index


def compute_noise_ratio(
    current_noise_std: float,
    baseline_noise_std: float,
) -> float:
    """Compute the noise degradation ratio.

    Parameters
    ----------
    current_noise_std : float
        Current noise standard deviation (e.g. in HU).
    baseline_noise_std : float
        Baseline noise standard deviation at commissioning.

    Returns
    -------
    float
        Ratio of current to baseline noise.  Values > 1.0 indicate
        degradation; values > 1.5 are concerning.

    Raises
    ------
    ValueError
        If ``baseline_noise_std`` is non-positive.
    """
    if baseline_noise_std <= 0.0:
        raise ValueError(
            f"baseline_noise_std must be positive, got {baseline_noise_std}"
        )
    return current_noise_std / baseline_noise_std


def compute_geometric_distortion_index(
    measured_distances: dict[str, float],
    expected_distances: dict[str, float],
) -> float:
    """Quantify geometric distortion from measured vs. expected distances.

    Computes the RMS of relative distance errors across all measurements.

    Parameters
    ----------
    measured_distances : dict[str, float]
        Measured distances keyed by measurement name (e.g. ``"diameter_x"``).
    expected_distances : dict[str, float]
        Expected (nominal) distances keyed by measurement name.

    Returns
    -------
    float
        Geometric distortion index >= 0.  Values near 0 indicate
        excellent geometric fidelity.
    """
    common_keys = set(measured_distances.keys()) & set(expected_distances.keys())
    if not common_keys:
        logger.warning(
            "No common distance measurements between measured and expected; "
            "returning 0.0."
        )
        return 0.0

    relative_errors_sq: list[float] = []
    for key in common_keys:
        expected = expected_distances[key]
        if abs(expected) < 1e-12:
            continue
        relative_error = (measured_distances[key] - expected) / expected
        relative_errors_sq.append(relative_error ** 2)

    if not relative_errors_sq:
        return 0.0

    rms = math.sqrt(sum(relative_errors_sq) / len(relative_errors_sq))
    return rms


# ---------------------------------------------------------------------------
# Aggregate feature computation
# ---------------------------------------------------------------------------

def compute_all_features(
    volume: np.ndarray,
    center: tuple[int, int],
    radius_px: int,
    current_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, float]:
    """Compute all six artifact signature features from a CT image slice.

    This convenience function calls each individual feature extractor and
    returns a unified dictionary suitable for :class:`ArtifactFeatures`
    construction.

    Parameters
    ----------
    volume : np.ndarray
        2-D CT image slice (H x W), typically in HU.
    center : tuple[int, int]
        Phantom centre coordinates ``(row, col)``.
    radius_px : int
        Phantom radius in pixels.
    current_metrics : dict
        Current QC measurement results.  Expected keys:

        * ``"insert_hu"`` -- dict[str, float] of current insert HU values
        * ``"noise_std"`` -- float, current noise standard deviation
        * ``"distances"`` -- dict[str, float] of measured geometric distances

    baseline_metrics : dict
        Baseline (commissioning) measurement results with the same keys.

    Returns
    -------
    dict[str, float]
        Feature dictionary with keys matching :class:`ArtifactFeatures` fields.
    """
    _require_numpy()

    # Image-based features
    inner_r = max(1, int(0.1 * radius_px))
    outer_r = max(inner_r + 1, int(0.95 * radius_px))

    ring = compute_ring_index(volume, center, inner_r, outer_r)
    cupping = compute_cupping_index(volume, center, radius_px)
    streak = compute_streak_index(volume, center, radius_px)

    # Metric-based features
    current_inserts = current_metrics.get("insert_hu", {})
    baseline_inserts = baseline_metrics.get("insert_hu", {})
    hu_drift = compute_hu_drift_index(current_inserts, baseline_inserts)

    current_noise = float(current_metrics.get("noise_std", 0.0))
    baseline_noise = float(baseline_metrics.get("noise_std", 1.0))
    noise = compute_noise_ratio(current_noise, baseline_noise) if baseline_noise > 0 else 1.0

    current_distances = current_metrics.get("distances", {})
    expected_distances = baseline_metrics.get("distances", {})
    geo_distortion = compute_geometric_distortion_index(
        current_distances, expected_distances,
    )

    return {
        "ring_index": ring,
        "cupping_index": cupping,
        "streak_index": streak,
        "hu_drift_index": hu_drift,
        "noise_ratio": noise,
        "geometric_distortion_index": geo_distortion,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_numpy() -> None:
    """Raise ``ImportError`` if numpy is not available."""
    if np is None:
        raise ImportError(
            "numpy is required for CT artifact feature extraction. "
            "Install it with: pip install numpy"
        )


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Minimal 2-D convolution without scipy dependency.

    Uses numpy-only implementation via stride tricks for small kernels.
    For production use with large images, consider replacing with
    ``scipy.signal.convolve2d`` or ``scipy.ndimage.convolve``.

    Parameters
    ----------
    image : np.ndarray
        2-D input image.
    kernel : np.ndarray
        Small 2-D convolution kernel (e.g. 3 x 3).

    Returns
    -------
    np.ndarray
        Convolved image (same shape as input, zero-padded at borders).
    """
    try:
        from scipy.ndimage import convolve
        return convolve(image, kernel, mode="constant", cval=0.0)
    except ImportError:
        pass

    # Fallback: manual convolution for 3x3 kernels
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
    output = np.zeros_like(image)
    for i in range(kh):
        for j in range(kw):
            output += kernel[i, j] * padded[i:i + image.shape[0], j:j + image.shape[1]]
    return output
