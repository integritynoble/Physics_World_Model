"""CT OperatorGraph for troubleshoot mode.

Provides a simplified CT forward model (Tier 1-2) for operator-correction
and reprojection-based diagnosis. Only activated when troubleshoot_mode=true.

Tier 1: Ideal parallel-beam Radon transform (analytical)
Tier 2: Fan-beam with geometric parameters (CoR, detector offsets)
Tier 3+: Cone-beam, scatter, beam hardening (future -- deferred)

The forward model is intentionally lightweight: it exists to compute
reprojection errors and estimate operator mismatches (center-of-rotation
offset, HU calibration drift, detector offsets), not to produce
clinical-quality reconstructions.

Usage
-----
>>> from pwm_core.clinical.ct.operator_graph import CTOperatorGraph, CTOperatorParams, CTGeometry
>>> geom = CTGeometry(num_angles=180, num_detectors=256, detector_spacing_mm=1.0)
>>> params = CTOperatorParams(geometry=geom)
>>> op = CTOperatorGraph(params)
>>> sinogram = op.forward(phantom_image)
>>> error = op.reprojection_error(phantom_image, measured_sinogram)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CTGeometry(BaseModel):
    """CT scanner geometry specification.

    Supports parallel-beam (Tier 1) when ``source_to_center_mm == 0``,
    and fan-beam (Tier 2) when ``source_to_center_mm > 0``.

    Attributes
    ----------
    num_angles : int
        Number of projection angles uniformly spaced over [0, pi).
    num_detectors : int
        Number of detector elements.
    detector_spacing_mm : float
        Center-to-center detector element spacing in mm.
    source_to_center_mm : float
        Source-to-isocenter distance. 0 = parallel beam (Tier 1).
    source_to_detector_mm : float
        Source-to-detector distance. 0 = parallel beam.
    center_of_rotation_offset_mm : float
        Lateral offset of the center of rotation from the detector center.
        Non-zero values model CoR misalignment.
    detector_offset_mm : float
        Lateral offset of the detector array from the nominal position.
    gantry_tilt_deg : float
        Gantry tilt angle in degrees. 0 = no tilt.
    """

    num_angles: int
    num_detectors: int
    detector_spacing_mm: float
    source_to_center_mm: float = 0.0
    source_to_detector_mm: float = 0.0
    center_of_rotation_offset_mm: float = 0.0
    detector_offset_mm: float = 0.0
    gantry_tilt_deg: float = 0.0

    @property
    def is_parallel_beam(self) -> bool:
        """True if geometry is parallel-beam (Tier 1)."""
        return self.source_to_center_mm == 0.0

    @property
    def is_fan_beam(self) -> bool:
        """True if geometry is fan-beam (Tier 2)."""
        return self.source_to_center_mm > 0.0


class CTOperatorParams(BaseModel):
    """Full CT operator parameterization.

    Combines scanner geometry with calibration and noise parameters.

    Attributes
    ----------
    geometry : CTGeometry
        Scanner geometry specification.
    hu_calibration_slope : float
        Slope of the HU calibration: HU = slope * raw + intercept.
    hu_calibration_intercept : float
        Intercept of the HU calibration.
    noise_std : float
        Standard deviation of additive Gaussian noise (in projection
        domain). 0 = noiseless.
    """

    geometry: CTGeometry
    hu_calibration_slope: float = 1.0
    hu_calibration_intercept: float = 0.0
    noise_std: float = 0.0


class MismatchEstimate(BaseModel):
    """Estimated mismatch for a single operator parameter.

    Produced by the diagnostic estimation routines (CoR offset estimation,
    HU calibration regression, etc.).

    Attributes
    ----------
    parameter_name : str
        Name of the estimated parameter (e.g., ``"center_of_rotation_offset_mm"``).
    estimated_value : float
        Estimated value of the parameter.
    confidence : float
        Confidence score between 0 and 1. Higher = more reliable estimate.
    method : str
        Name of the estimation method used (e.g., ``"sinogram_symmetry"``,
        ``"linear_regression"``).
    """

    parameter_name: str
    estimated_value: float
    confidence: float = Field(ge=0.0, le=1.0)
    method: str


# ---------------------------------------------------------------------------
# CTOperatorGraph
# ---------------------------------------------------------------------------


class CTOperatorGraph:
    """Simplified CT forward model for troubleshoot-mode diagnosis.

    Implements Tier 1 (parallel-beam Radon) and Tier 2 (fan-beam) forward
    projection, backprojection, and parameter estimation routines. Used for
    computing reprojection errors and estimating operator mismatches.

    Parameters
    ----------
    params : CTOperatorParams
        Complete operator parameterization including geometry and calibration.

    Notes
    -----
    This is a diagnostic tool, not a clinical reconstruction engine. The
    forward model is accurate enough for mismatch estimation (CoR offset,
    HU drift) but deliberately omits scatter, beam hardening, and other
    higher-order effects.
    """

    def __init__(self, params: CTOperatorParams) -> None:
        self.params = params
        self._angles = np.linspace(
            0, np.pi, params.geometry.num_angles, endpoint=False
        )

    # ------------------------------------------------------------------
    # Forward projection
    # ------------------------------------------------------------------

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Apply the CT forward projection (image -> sinogram).

        Dispatches to parallel-beam Radon (Tier 1) or fan-beam projection
        (Tier 2) based on the geometry configuration.

        Parameters
        ----------
        image : np.ndarray
            2D image array (N x N) representing the attenuation map.

        Returns
        -------
        np.ndarray
            Sinogram of shape (num_angles, num_detectors).
        """
        image = np.asarray(image, dtype=np.float64)

        if self.params.geometry.is_parallel_beam:
            sinogram = self._radon_transform(image, self._angles)
        else:
            sinogram = self._fan_beam_forward(image)

        # Apply noise if configured
        if self.params.noise_std > 0:
            rng = np.random.default_rng()
            sinogram = sinogram + rng.normal(
                0, self.params.noise_std, sinogram.shape
            )

        return sinogram

    # ------------------------------------------------------------------
    # Adjoint (backprojection)
    # ------------------------------------------------------------------

    def adjoint(self, sinogram: np.ndarray) -> np.ndarray:
        """Apply the adjoint operator (sinogram -> image via backprojection).

        This is the unfiltered backprojection (A^T), not a reconstruction.
        It is the mathematical adjoint of the forward projection and is
        used for iterative algorithms and reprojection diagnostics.

        Parameters
        ----------
        sinogram : np.ndarray
            Sinogram of shape (num_angles, num_detectors).

        Returns
        -------
        np.ndarray
            Backprojected image (N x N).
        """
        sinogram = np.asarray(sinogram, dtype=np.float64)
        return self._backproject(sinogram, self._angles)

    # ------------------------------------------------------------------
    # Reprojection error
    # ------------------------------------------------------------------

    def reprojection_error(
        self, image: np.ndarray, measured_sinogram: np.ndarray
    ) -> float:
        """Compute the normalized reprojection error.

        Calculates ||A*x - y||^2 / ||y||^2, where A is the forward
        operator, x is the image, and y is the measured sinogram.

        A large reprojection error indicates operator mismatch (wrong
        geometry, miscalibration, etc.).

        Parameters
        ----------
        image : np.ndarray
            2D image (N x N).
        measured_sinogram : np.ndarray
            Measured sinogram (num_angles x num_detectors).

        Returns
        -------
        float
            Normalized squared error. 0.0 = perfect match.
        """
        image = np.asarray(image, dtype=np.float64)
        measured_sinogram = np.asarray(measured_sinogram, dtype=np.float64)

        predicted = self.forward(image)

        # Handle shape mismatch gracefully
        if predicted.shape != measured_sinogram.shape:
            logger.warning(
                "Shape mismatch: predicted %s vs measured %s. "
                "Attempting to crop/pad to match.",
                predicted.shape, measured_sinogram.shape,
            )
            predicted, measured_sinogram = _match_shapes(
                predicted, measured_sinogram
            )

        residual_norm_sq = float(np.sum((predicted - measured_sinogram) ** 2))
        measured_norm_sq = float(np.sum(measured_sinogram ** 2))

        if measured_norm_sq < 1e-30:
            logger.warning("Measured sinogram is effectively zero.")
            return float("inf")

        return residual_norm_sq / measured_norm_sq

    # ------------------------------------------------------------------
    # CoR offset estimation
    # ------------------------------------------------------------------

    def estimate_cor_offset(self, sinogram: np.ndarray) -> MismatchEstimate:
        """Estimate center-of-rotation offset from sinogram symmetry.

        Uses the 180-degree conjugate-ray consistency condition: for
        parallel-beam geometry, projection at angle theta and at
        (theta + pi) should be mirror images. Deviation from this
        symmetry indicates a CoR offset.

        The offset is estimated as the shift that maximizes the
        cross-correlation between conjugate projection pairs.

        Parameters
        ----------
        sinogram : np.ndarray
            Measured sinogram (num_angles x num_detectors).

        Returns
        -------
        MismatchEstimate
            Estimated CoR offset in mm with confidence score.
        """
        sinogram = np.asarray(sinogram, dtype=np.float64)
        num_angles, num_detectors = sinogram.shape
        half_angles = num_angles // 2

        if half_angles < 2:
            return MismatchEstimate(
                parameter_name="center_of_rotation_offset_mm",
                estimated_value=0.0,
                confidence=0.0,
                method="sinogram_symmetry",
            )

        # Accumulate cross-correlation-based shift estimates
        shifts: list[float] = []
        weights: list[float] = []

        for i in range(half_angles):
            conjugate_idx = i + half_angles
            if conjugate_idx >= num_angles:
                break

            proj_0 = sinogram[i]
            proj_180 = sinogram[conjugate_idx][::-1]  # flip for conjugate

            # Cross-correlation via FFT to find sub-pixel shift
            shift = _cross_correlation_shift(proj_0, proj_180)
            shifts.append(shift)

            # Weight by signal energy (stronger projections give better estimates)
            energy = float(np.sum(proj_0 ** 2) + np.sum(proj_180 ** 2))
            weights.append(energy)

        if not shifts:
            return MismatchEstimate(
                parameter_name="center_of_rotation_offset_mm",
                estimated_value=0.0,
                confidence=0.0,
                method="sinogram_symmetry",
            )

        shifts_arr = np.array(shifts)
        weights_arr = np.array(weights)

        # Normalize weights
        total_weight = weights_arr.sum()
        if total_weight > 0:
            weights_arr = weights_arr / total_weight

        # Weighted mean shift (in detector pixels)
        mean_shift_px = float(np.sum(shifts_arr * weights_arr))

        # Convert pixel shift to mm (shift is half the CoR offset)
        cor_offset_mm = mean_shift_px * self.params.geometry.detector_spacing_mm / 2.0

        # Confidence: based on consistency of shift estimates
        if len(shifts) >= 3:
            std_shift = float(np.std(shifts_arr))
            # Lower std -> higher confidence
            confidence = float(np.clip(1.0 - std_shift / (abs(mean_shift_px) + 0.5), 0.0, 1.0))
        else:
            confidence = 0.3  # Low confidence with few samples

        return MismatchEstimate(
            parameter_name="center_of_rotation_offset_mm",
            estimated_value=round(cor_offset_mm, 4),
            confidence=round(confidence, 4),
            method="sinogram_symmetry",
        )

    # ------------------------------------------------------------------
    # HU calibration estimation
    # ------------------------------------------------------------------

    def estimate_hu_calibration(
        self,
        phantom_rois: dict[str, float],
        expected_hu: dict[str, float],
    ) -> tuple[MismatchEstimate, MismatchEstimate]:
        """Estimate HU calibration slope and intercept from known inserts.

        Performs a least-squares linear fit of measured ROI values against
        known expected HU values for phantom inserts (e.g., ACR phantom
        modules: water, air, bone, polyethylene, acrylic).

        Parameters
        ----------
        phantom_rois : dict[str, float]
            Measured mean HU values for each phantom insert.
            Keys are insert names, values are measured HU.
        expected_hu : dict[str, float]
            Expected (reference) HU values for the same inserts.
            Keys must overlap with ``phantom_rois``.

        Returns
        -------
        tuple[MismatchEstimate, MismatchEstimate]
            (slope_estimate, intercept_estimate). Ideal calibration is
            slope=1.0, intercept=0.0.

        Raises
        ------
        ValueError
            If fewer than 2 matching insert names are found.
        """
        # Find common inserts
        common_keys = sorted(
            set(phantom_rois.keys()) & set(expected_hu.keys())
        )

        if len(common_keys) < 2:
            raise ValueError(
                f"Need at least 2 matching inserts for calibration fit, "
                f"got {len(common_keys)}. "
                f"phantom_rois keys: {sorted(phantom_rois.keys())}, "
                f"expected_hu keys: {sorted(expected_hu.keys())}"
            )

        measured = np.array([phantom_rois[k] for k in common_keys], dtype=np.float64)
        expected = np.array([expected_hu[k] for k in common_keys], dtype=np.float64)

        # Least-squares: measured = slope * expected + intercept
        # Use numpy polyfit (degree 1): measured = slope * expected + intercept
        coeffs = np.polyfit(expected, measured, deg=1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])

        # Compute R-squared for confidence
        predicted = slope * expected + intercept
        ss_res = float(np.sum((measured - predicted) ** 2))
        ss_tot = float(np.sum((measured - np.mean(measured)) ** 2))

        if ss_tot > 1e-30:
            r_squared = 1.0 - ss_res / ss_tot
        else:
            r_squared = 0.0

        # Confidence: R^2 scaled — good linear fit = high confidence
        confidence = float(np.clip(r_squared, 0.0, 1.0))

        slope_estimate = MismatchEstimate(
            parameter_name="hu_calibration_slope",
            estimated_value=round(slope, 6),
            confidence=round(confidence, 4),
            method="linear_regression",
        )

        intercept_estimate = MismatchEstimate(
            parameter_name="hu_calibration_intercept",
            estimated_value=round(intercept, 4),
            confidence=round(confidence, 4),
            method="linear_regression",
        )

        logger.info(
            "HU calibration estimated: slope=%.4f, intercept=%.2f, "
            "R^2=%.4f (n=%d inserts)",
            slope, intercept, r_squared, len(common_keys),
        )

        return slope_estimate, intercept_estimate

    # ------------------------------------------------------------------
    # Operator correction
    # ------------------------------------------------------------------

    def correct_operator(
        self, estimates: list[MismatchEstimate]
    ) -> CTOperatorParams:
        """Return corrected operator params based on mismatch estimates.

        Applies each estimate to the corresponding parameter in the current
        params, producing a new ``CTOperatorParams`` with the corrected
        values. Original params are not mutated.

        Parameters
        ----------
        estimates : list[MismatchEstimate]
            List of parameter mismatch estimates. Only estimates with
            confidence > 0.1 are applied.

        Returns
        -------
        CTOperatorParams
            New params with corrections applied.
        """
        # Deep copy current params
        corrected_geom = self.params.geometry.model_copy(deep=True)
        corrected_slope = self.params.hu_calibration_slope
        corrected_intercept = self.params.hu_calibration_intercept
        corrected_noise = self.params.noise_std

        min_confidence = 0.1

        for est in estimates:
            if est.confidence < min_confidence:
                logger.info(
                    "Skipping low-confidence estimate: %s = %.4f (conf=%.2f)",
                    est.parameter_name, est.estimated_value, est.confidence,
                )
                continue

            name = est.parameter_name

            if name == "center_of_rotation_offset_mm":
                corrected_geom = corrected_geom.model_copy(
                    update={"center_of_rotation_offset_mm": est.estimated_value}
                )
            elif name == "detector_offset_mm":
                corrected_geom = corrected_geom.model_copy(
                    update={"detector_offset_mm": est.estimated_value}
                )
            elif name == "gantry_tilt_deg":
                corrected_geom = corrected_geom.model_copy(
                    update={"gantry_tilt_deg": est.estimated_value}
                )
            elif name == "hu_calibration_slope":
                corrected_slope = est.estimated_value
            elif name == "hu_calibration_intercept":
                corrected_intercept = est.estimated_value
            elif name == "noise_std":
                corrected_noise = est.estimated_value
            else:
                logger.warning(
                    "Unknown parameter in mismatch estimate: %s", name
                )

        corrected_params = CTOperatorParams(
            geometry=corrected_geom,
            hu_calibration_slope=corrected_slope,
            hu_calibration_intercept=corrected_intercept,
            noise_std=corrected_noise,
        )

        logger.info("Operator corrected with %d estimates.", len(estimates))
        return corrected_params

    # ------------------------------------------------------------------
    # Tier 1: Parallel-beam Radon transform
    # ------------------------------------------------------------------

    def _radon_transform(
        self, image: np.ndarray, angles: np.ndarray
    ) -> np.ndarray:
        """Parallel-beam Radon transform (Tier 1).

        Tries to use ``skimage.transform.radon`` for speed and accuracy.
        Falls back to a rotation-based numpy implementation if scikit-image
        is not available.

        Parameters
        ----------
        image : np.ndarray
            2D image (N x N).
        angles : np.ndarray
            Projection angles in radians.

        Returns
        -------
        np.ndarray
            Sinogram of shape (len(angles), num_detectors).
        """
        angles_deg = np.degrees(angles)
        num_detectors = self.params.geometry.num_detectors
        cor_offset_px = (
            self.params.geometry.center_of_rotation_offset_mm
            / self.params.geometry.detector_spacing_mm
        )

        # Try scikit-image first
        try:
            from skimage.transform import radon as skimage_radon

            # skimage.radon expects angles in degrees and handles its own
            # detector sizing. We compute with its native output, then
            # resample to our detector count.
            sinogram_raw = skimage_radon(
                image, theta=angles_deg, circle=True
            )
            # sinogram_raw shape: (detector_native, num_angles) -- transpose
            # to (num_angles, detector_native)
            sinogram_raw = sinogram_raw.T

            # Resample to target detector count if needed
            sinogram = _resample_detectors(
                sinogram_raw, num_detectors, cor_offset_px
            )

            return sinogram

        except ImportError:
            logger.debug(
                "scikit-image not available; using numpy Radon fallback."
            )

        # Fallback: rotation-based Radon using scipy.ndimage
        return self._radon_numpy(image, angles, num_detectors, cor_offset_px)

    def _radon_numpy(
        self,
        image: np.ndarray,
        angles: np.ndarray,
        num_detectors: int,
        cor_offset_px: float,
    ) -> np.ndarray:
        """Numpy/scipy fallback for parallel-beam Radon transform.

        Rotates the image by each angle and sums along columns to form
        each projection line. Uses scipy.ndimage.rotate for sub-pixel
        accuracy.

        Parameters
        ----------
        image : np.ndarray
            2D image (N x N).
        angles : np.ndarray
            Projection angles in radians.
        num_detectors : int
            Number of detector elements in the output sinogram.
        cor_offset_px : float
            Center-of-rotation offset in pixels.

        Returns
        -------
        np.ndarray
            Sinogram of shape (len(angles), num_detectors).
        """
        from scipy.ndimage import rotate as ndimage_rotate, shift as ndimage_shift

        n = image.shape[0]
        sinogram = np.zeros((len(angles), num_detectors), dtype=np.float64)

        # Pad image to avoid clipping during rotation
        pad_size = int(np.ceil(n * (np.sqrt(2) - 1) / 2)) + 2
        padded = np.pad(image, pad_size, mode="constant", constant_values=0)

        for i, angle in enumerate(angles):
            angle_deg = float(np.degrees(angle))

            # Apply CoR offset as a shift before rotation
            if abs(cor_offset_px) > 1e-6:
                shifted = ndimage_shift(
                    padded, [0, cor_offset_px], order=1, mode="constant"
                )
            else:
                shifted = padded

            # Rotate and sum along columns (axis=0)
            rotated = ndimage_rotate(
                shifted, angle_deg, reshape=False, order=1, mode="constant"
            )
            projection = np.sum(rotated, axis=0)

            # Resample to target detector count
            sinogram[i] = _resample_1d(projection, num_detectors)

        return sinogram

    # ------------------------------------------------------------------
    # Tier 2: Fan-beam forward projection
    # ------------------------------------------------------------------

    def _fan_beam_forward(self, image: np.ndarray) -> np.ndarray:
        """Fan-beam forward projection (Tier 2).

        Implements a simple ray-driven fan-beam projector. For each angle
        and detector element, computes the ray path from source to
        detector and accumulates attenuation values along the ray using
        bilinear interpolation.

        Parameters
        ----------
        image : np.ndarray
            2D image (N x N).

        Returns
        -------
        np.ndarray
            Sinogram of shape (num_angles, num_detectors).
        """
        geom = self.params.geometry
        num_angles = geom.num_angles
        num_detectors = geom.num_detectors
        dsd = geom.source_to_detector_mm
        dsc = geom.source_to_center_mm
        det_spacing = geom.detector_spacing_mm
        cor_offset = geom.center_of_rotation_offset_mm
        det_offset = geom.detector_offset_mm

        n = image.shape[0]
        pixel_size = det_spacing  # Assume pixel size = detector spacing

        sinogram = np.zeros((num_angles, num_detectors), dtype=np.float64)

        # Detector element positions (centered, with offset)
        det_positions = (
            (np.arange(num_detectors) - (num_detectors - 1) / 2.0)
            * det_spacing
            + det_offset
        )

        for i, angle in enumerate(self._angles):
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Source position
            sx = -dsc * sin_a + cor_offset * cos_a
            sy = dsc * cos_a + cor_offset * sin_a

            for j, det_pos in enumerate(det_positions):
                # Detector element position
                dx = (dsd - dsc) * sin_a + det_pos * cos_a
                dy = -(dsd - dsc) * cos_a + det_pos * sin_a

                # Ray from source to detector
                sinogram[i, j] = _trace_ray(
                    image, sx, sy, dx, dy, n, pixel_size
                )

        return sinogram

    # ------------------------------------------------------------------
    # Backprojection (adjoint)
    # ------------------------------------------------------------------

    def _backproject(
        self, sinogram: np.ndarray, angles: np.ndarray
    ) -> np.ndarray:
        """Filtered backprojection (adjoint operation).

        Applies a ramp filter in the Fourier domain, then backprojects
        each filtered projection onto the image grid.

        For parallel-beam geometry, this implements standard FBP.
        For fan-beam, it uses a simplified parallel-beam approximation
        (sufficient for diagnostic purposes).

        Parameters
        ----------
        sinogram : np.ndarray
            Sinogram of shape (num_angles, num_detectors).
        angles : np.ndarray
            Projection angles in radians.

        Returns
        -------
        np.ndarray
            Reconstructed image (N x N) where N = num_detectors.
        """
        from scipy.ndimage import rotate as ndimage_rotate

        num_angles, num_detectors = sinogram.shape
        n = num_detectors

        # Apply ramp filter to each projection
        filtered_sinogram = _apply_ramp_filter(sinogram)

        # Backproject: smear each filtered projection across the image
        image = np.zeros((n, n), dtype=np.float64)

        for i, angle in enumerate(angles):
            angle_deg = float(np.degrees(angle))
            projection = filtered_sinogram[i]

            # Create a 2D image by replicating the projection along rows
            proj_image = np.tile(projection, (n, 1))

            # Rotate by the negative angle and accumulate
            rotated = ndimage_rotate(
                proj_image, -angle_deg, reshape=False, order=1, mode="constant"
            )
            image += rotated

        # Normalize by number of angles
        image *= np.pi / (2 * num_angles)

        return image


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _cross_correlation_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the sub-pixel shift between two 1D signals via FFT cross-correlation.

    Returns the shift (in pixels) that maximizes the correlation of b
    with respect to a. Positive shift means b is shifted to the right
    relative to a.

    Parameters
    ----------
    a, b : np.ndarray
        1D signals of the same length.

    Returns
    -------
    float
        Estimated shift in pixels (sub-pixel via parabolic interpolation).
    """
    n = len(a)
    if n == 0:
        return 0.0

    # Zero-mean normalization
    a = a - np.mean(a)
    b = b - np.mean(b)

    # Cross-correlation via FFT
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    cross = np.fft.ifft(fa * np.conj(fb)).real

    # Find peak
    peak_idx = int(np.argmax(cross))

    # Parabolic interpolation for sub-pixel accuracy
    if 0 < peak_idx < n - 1:
        y_m1 = cross[peak_idx - 1]
        y_0 = cross[peak_idx]
        y_p1 = cross[peak_idx + 1]
        denom = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
        if abs(denom) > 1e-12:
            delta = (y_m1 - y_p1) / denom
        else:
            delta = 0.0
    else:
        delta = 0.0

    shift = float(peak_idx) + delta
    # Wrap around: if shift > n/2, it's a negative shift
    if shift > n / 2:
        shift -= n

    return shift


def _resample_1d(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a 1D signal to a target length using linear interpolation.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal.
    target_length : int
        Desired output length.

    Returns
    -------
    np.ndarray
        Resampled signal of length ``target_length``.
    """
    n = len(signal)
    if n == target_length:
        return signal.copy()

    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, signal)


def _resample_detectors(
    sinogram: np.ndarray,
    target_detectors: int,
    cor_offset_px: float = 0.0,
) -> np.ndarray:
    """Resample sinogram detectors and apply CoR offset.

    Parameters
    ----------
    sinogram : np.ndarray
        Input sinogram (num_angles x native_detectors).
    target_detectors : int
        Target number of detector elements.
    cor_offset_px : float
        Center-of-rotation offset in pixels (applied as a lateral shift).

    Returns
    -------
    np.ndarray
        Resampled sinogram (num_angles x target_detectors).
    """
    num_angles, native_detectors = sinogram.shape
    result = np.zeros((num_angles, target_detectors), dtype=np.float64)

    for i in range(num_angles):
        resampled = _resample_1d(sinogram[i], target_detectors)
        if abs(cor_offset_px) > 1e-6:
            # Apply sub-pixel shift via FFT
            freq = np.fft.fftfreq(target_detectors)
            phase_shift = np.exp(-2j * np.pi * freq * cor_offset_px)
            resampled = np.fft.ifft(
                np.fft.fft(resampled) * phase_shift
            ).real
        result[i] = resampled

    return result


def _trace_ray(
    image: np.ndarray,
    sx: float, sy: float,
    dx: float, dy: float,
    n: int,
    pixel_size: float,
) -> float:
    """Trace a ray through a 2D image and accumulate attenuation.

    Uses Siddon's algorithm (simplified) for ray-driven integration
    through the pixel grid.

    Parameters
    ----------
    image : np.ndarray
        2D image (N x N).
    sx, sy : float
        Source position in mm (world coordinates).
    dx, dy : float
        Detector element position in mm (world coordinates).
    n : int
        Image dimension.
    pixel_size : float
        Size of each pixel in mm.

    Returns
    -------
    float
        Integrated attenuation along the ray.
    """
    # Convert world coordinates to pixel coordinates
    # Image center is at (n/2, n/2) in pixel coords
    half_n = n / 2.0

    sx_px = sx / pixel_size + half_n
    sy_px = sy / pixel_size + half_n
    dx_px = dx / pixel_size + half_n
    dy_px = dy / pixel_size + half_n

    # Ray direction
    ray_dx = dx_px - sx_px
    ray_dy = dy_px - sy_px
    ray_len = np.sqrt(ray_dx ** 2 + ray_dy ** 2)

    if ray_len < 1e-10:
        return 0.0

    # Normalize direction
    ray_dx /= ray_len
    ray_dy /= ray_len

    # Step along the ray (simple sampling approach)
    num_steps = int(np.ceil(ray_len * 1.5))
    step_size = ray_len / max(num_steps, 1)

    total = 0.0
    for s in range(num_steps):
        t = (s + 0.5) * step_size
        px = sx_px + ray_dx * t
        py = sy_px + ray_dy * t

        # Bilinear interpolation
        ix = int(np.floor(px))
        iy = int(np.floor(py))

        if 0 <= ix < n - 1 and 0 <= iy < n - 1:
            fx = px - ix
            fy = py - iy
            val = (
                image[iy, ix] * (1 - fx) * (1 - fy)
                + image[iy, ix + 1] * fx * (1 - fy)
                + image[iy + 1, ix] * (1 - fx) * fy
                + image[iy + 1, ix + 1] * fx * fy
            )
            total += val * step_size * pixel_size

    return total


def _apply_ramp_filter(sinogram: np.ndarray) -> np.ndarray:
    """Apply a ramp (Ram-Lak) filter to each projection in the sinogram.

    The ramp filter is applied in the Fourier domain. A Hann window is
    used to suppress high-frequency noise.

    Parameters
    ----------
    sinogram : np.ndarray
        Sinogram of shape (num_angles, num_detectors).

    Returns
    -------
    np.ndarray
        Filtered sinogram.
    """
    num_angles, num_detectors = sinogram.shape

    # Pad to next power of 2 for efficient FFT
    pad_len = int(2 ** np.ceil(np.log2(2 * num_detectors)))
    padded = np.zeros((num_angles, pad_len), dtype=np.float64)
    padded[:, :num_detectors] = sinogram

    # Ramp filter in frequency domain
    freqs = np.fft.fftfreq(pad_len)
    ramp = np.abs(freqs) * 2.0  # Ramp

    # Apply Hann window to suppress noise
    hann = 0.5 * (1.0 + np.cos(2.0 * np.pi * freqs))
    filt = ramp * hann

    # Apply filter to each projection
    filtered = np.zeros_like(padded)
    for i in range(num_angles):
        ft = np.fft.fft(padded[i])
        filtered[i] = np.fft.ifft(ft * filt).real

    return filtered[:, :num_detectors]


def _match_shapes(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Crop or pad two 2D arrays to the same shape.

    Takes the minimum shape along each axis and center-crops both arrays.

    Parameters
    ----------
    a, b : np.ndarray
        Input 2D arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cropped arrays with matching shapes.
    """
    min_rows = min(a.shape[0], b.shape[0])
    min_cols = min(a.shape[1], b.shape[1])

    # Center crop
    a_row_start = (a.shape[0] - min_rows) // 2
    a_col_start = (a.shape[1] - min_cols) // 2
    b_row_start = (b.shape[0] - min_rows) // 2
    b_col_start = (b.shape[1] - min_cols) // 2

    a_cropped = a[a_row_start:a_row_start + min_rows, a_col_start:a_col_start + min_cols]
    b_cropped = b[b_row_start:b_row_start + min_rows, b_col_start:b_col_start + min_cols]

    return a_cropped, b_cropped
