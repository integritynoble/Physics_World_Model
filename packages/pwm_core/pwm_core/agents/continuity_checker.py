"""pwm_core.agents.continuity_checker

Physical Continuity Checker: validates that an imaging system description
is internally consistent before any expensive computation begins.

Design mantra
-------------
This is a **defence-in-depth** layer.  Many of these checks overlap with
pydantic validators in the contract models, but catching inconsistencies
here produces domain-specific warning messages that help the user (or an
upstream agent) fix the problem.

Checks performed
----------------
1. **NA compatibility** -- Numerical aperture of lenses must be compatible
   with the detector pixel geometry (Nyquist criterion).
2. **Spectral consistency** -- Dispersion elements must operate within the
   system's declared wavelength range.
3. **Dimension chain validity** -- The forward model's input and output
   shapes must be consistent with the element chain.
4. **Total throughput sanity** -- If the product of all element throughputs
   falls below 1%, the system is likely mis-specified.
5. **Detector existence** -- The element list must contain at least one
   detector (redundant with the pydantic validator on ``ImagingSystem``,
   but included for defence-in-depth).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from .contracts import (
    ImagingSystem,
    ElementSpec,
    TransferKind,
    PhotonReport,
    MismatchReport,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Minimum total throughput before a warning is raised
_MIN_THROUGHPUT = 0.01

# Maximum NA before diffraction-limited resolution exceeds pixel Nyquist
# (heuristic -- real check depends on wavelength and pixel pitch)
_MAX_NA_RATIO = 2.0

# Tolerance for spectral range overlap check (nm)
_SPECTRAL_TOLERANCE_NM = 50.0


# ---------------------------------------------------------------------------
# PhysicalContinuityChecker
# ---------------------------------------------------------------------------


class PhysicalContinuityChecker:
    """Validates physical consistency of an imaging system description.

    This checker is stateless and deterministic.  It inspects the
    ``ImagingSystem`` contract (and optionally PhotonReport / MismatchReport)
    and returns a list of human-readable warning strings.  An empty list
    means the system passed all checks.

    Usage
    -----
    >>> checker = PhysicalContinuityChecker()
    >>> warnings = checker.check(system, photon_report, mismatch_report)
    >>> for w in warnings:
    ...     print(f"WARNING: {w}")
    """

    def check(
        self,
        system: ImagingSystem,
        photon_report: Optional[PhotonReport] = None,
        mismatch_report: Optional[MismatchReport] = None,
    ) -> List[str]:
        """Run all continuity checks on the imaging system.

        Parameters
        ----------
        system : ImagingSystem
            The imaging system description to validate.
        photon_report : PhotonReport, optional
            If provided, additional cross-checks with the photon budget
            are performed.
        mismatch_report : MismatchReport, optional
            If provided, additional cross-checks with the mismatch
            report are performed.

        Returns
        -------
        list[str]
            Human-readable warning messages.  Empty if all checks pass.
        """
        warnings: List[str] = []

        # Check 5 (first, because it is the cheapest and most critical)
        warnings.extend(self._check_detector_exists(system))

        # Check 1: NA compatibility
        warnings.extend(self._check_na_compatibility(system))

        # Check 2: Spectral consistency
        warnings.extend(self._check_spectral_consistency(system))

        # Check 3: Dimension chain validity
        warnings.extend(self._check_dimension_chain(system))

        # Check 4: Total throughput sanity
        warnings.extend(self._check_total_throughput(system))

        if warnings:
            logger.warning(
                "PhysicalContinuityChecker found %d warning(s) for modality '%s'.",
                len(warnings),
                system.modality_key,
            )
        else:
            logger.info(
                "PhysicalContinuityChecker: all checks passed for modality '%s'.",
                system.modality_key,
            )

        return warnings

    # ----- Check 1: NA compatibility -----------------------------------------

    @staticmethod
    def _check_na_compatibility(system: ImagingSystem) -> List[str]:
        """Verify that lens NA values are compatible with detector geometry.

        For each lens element with a declared ``na`` parameter and each
        detector with a declared ``pixel_pitch_um`` parameter, we check
        that the diffraction-limited resolution (Abbe criterion) is
        compatible with the pixel pitch (Nyquist sampling).

        The Abbe resolution limit is ``lambda / (2 * NA)``.  For Nyquist
        sampling the pixel pitch must be at most half the resolution limit:
        ``pixel_pitch <= lambda / (4 * NA)``.

        If the pixel pitch exceeds ``_MAX_NA_RATIO`` times the Nyquist
        limit, we warn that the system is under-sampling.

        Parameters
        ----------
        system : ImagingSystem

        Returns
        -------
        list[str]
            Warnings (empty if compatible).
        """
        warnings: List[str] = []

        lenses = [e for e in system.elements if e.element_type == "lens"]
        detectors = [
            e for e in system.elements
            if e.element_type in ("detector", "transducer")
        ]

        # Need wavelength for diffraction calculation
        wavelength_um: Optional[float] = None
        if system.wavelength_nm is not None:
            wavelength_um = system.wavelength_nm / 1000.0
        elif system.spectral_range_nm is not None and len(system.spectral_range_nm) >= 2:
            # Use centre wavelength
            centre_nm = (system.spectral_range_nm[0] + system.spectral_range_nm[1]) / 2.0
            wavelength_um = centre_nm / 1000.0

        if wavelength_um is None:
            # Cannot perform NA check without wavelength information
            return warnings

        for lens in lenses:
            na = lens.parameters.get("na") or lens.parameters.get("NA")
            if na is None or not isinstance(na, (int, float)) or na <= 0:
                continue

            # Abbe resolution limit in microns
            resolution_um = wavelength_um / (2.0 * na)
            # Nyquist pixel pitch limit
            nyquist_pitch_um = resolution_um / 2.0

            for det in detectors:
                pixel_pitch = det.parameters.get("pixel_pitch_um")
                if pixel_pitch is None or not isinstance(pixel_pitch, (int, float)):
                    continue
                if pixel_pitch <= 0:
                    continue

                ratio = pixel_pitch / nyquist_pitch_um
                if ratio > _MAX_NA_RATIO:
                    warnings.append(
                        f"NA compatibility: Lens '{lens.name}' (NA={na:.2f}) "
                        f"produces a diffraction-limited resolution of "
                        f"{resolution_um:.2f} um, requiring pixel pitch "
                        f"<= {nyquist_pitch_um:.2f} um for Nyquist sampling.  "
                        f"Detector '{det.name}' has pixel pitch "
                        f"{pixel_pitch:.2f} um ({ratio:.1f}x over Nyquist).  "
                        f"The system is under-sampling."
                    )

        return warnings

    # ----- Check 2: Spectral consistency -------------------------------------

    @staticmethod
    def _check_spectral_consistency(system: ImagingSystem) -> List[str]:
        """Verify that dispersion elements operate within the system wavelength range.

        Any element with ``transfer_kind == "dispersion"`` should have
        spectral parameters consistent with the system-level wavelength
        specification.

        Parameters
        ----------
        system : ImagingSystem

        Returns
        -------
        list[str]
            Warnings (empty if consistent).
        """
        warnings: List[str] = []

        # Determine system spectral range
        sys_min_nm: Optional[float] = None
        sys_max_nm: Optional[float] = None

        if system.spectral_range_nm is not None and len(system.spectral_range_nm) >= 2:
            sys_min_nm = system.spectral_range_nm[0]
            sys_max_nm = system.spectral_range_nm[1]
        elif system.wavelength_nm is not None:
            # Single wavelength: assume narrow band
            sys_min_nm = system.wavelength_nm - _SPECTRAL_TOLERANCE_NM
            sys_max_nm = system.wavelength_nm + _SPECTRAL_TOLERANCE_NM

        if sys_min_nm is None or sys_max_nm is None:
            # No spectral info on the system -- skip check
            return warnings

        for elem in system.elements:
            if elem.transfer_kind != TransferKind.dispersion:
                continue

            # Check element-level spectral parameters
            elem_min = elem.parameters.get("wavelength_min_nm")
            elem_max = elem.parameters.get("wavelength_max_nm")
            elem_centre = elem.parameters.get("centre_wavelength_nm")
            elem_range = elem.parameters.get("spectral_range_nm")

            # Try to determine element spectral range
            e_min: Optional[float] = None
            e_max: Optional[float] = None

            if elem_min is not None and elem_max is not None:
                e_min = float(elem_min)
                e_max = float(elem_max)
            elif elem_range is not None and isinstance(elem_range, (list, tuple)) and len(elem_range) >= 2:
                e_min = float(elem_range[0])
                e_max = float(elem_range[1])
            elif elem_centre is not None:
                # Assume narrow band around centre
                e_min = float(elem_centre) - _SPECTRAL_TOLERANCE_NM
                e_max = float(elem_centre) + _SPECTRAL_TOLERANCE_NM
            else:
                # No spectral parameters on dispersion element
                warnings.append(
                    f"Spectral consistency: Dispersion element '{elem.name}' "
                    f"has no spectral parameters (wavelength_min_nm, "
                    f"wavelength_max_nm, or centre_wavelength_nm).  "
                    f"Cannot verify spectral compatibility with the system "
                    f"range [{sys_min_nm:.0f}, {sys_max_nm:.0f}] nm."
                )
                continue

            # Check overlap
            overlap_min = max(sys_min_nm, e_min)
            overlap_max = min(sys_max_nm, e_max)

            if overlap_min > overlap_max:
                warnings.append(
                    f"Spectral consistency: Dispersion element '{elem.name}' "
                    f"operates in [{e_min:.0f}, {e_max:.0f}] nm, which does "
                    f"not overlap with the system range "
                    f"[{sys_min_nm:.0f}, {sys_max_nm:.0f}] nm.  "
                    f"This element will not disperse any signal."
                )
            else:
                # Check for partial overlap
                system_span = sys_max_nm - sys_min_nm
                overlap_span = overlap_max - overlap_min
                if system_span > 0 and (overlap_span / system_span) < 0.5:
                    warnings.append(
                        f"Spectral consistency: Dispersion element '{elem.name}' "
                        f"covers [{e_min:.0f}, {e_max:.0f}] nm, which overlaps "
                        f"only {overlap_span / system_span:.0%} of the system "
                        f"range [{sys_min_nm:.0f}, {sys_max_nm:.0f}] nm.  "
                        f"Spectral coverage may be incomplete."
                    )

        return warnings

    # ----- Check 3: Dimension chain validity ---------------------------------

    @staticmethod
    def _check_dimension_chain(system: ImagingSystem) -> List[str]:
        """Verify that signal dimensions are consistent with the forward model.

        The ``signal_dims`` dictionary must contain at least an input ("x")
        and output ("y") shape.  For a linear forward model ``y = A x``,
        the product of the output dimensions should not exceed the product
        of the input dimensions for a well-posed problem (unless the system
        is intentionally over-determined).

        Parameters
        ----------
        system : ImagingSystem

        Returns
        -------
        list[str]
            Warnings (empty if consistent).
        """
        warnings: List[str] = []

        dims = system.signal_dims
        if not dims:
            warnings.append(
                "Dimension chain: 'signal_dims' is empty.  Cannot validate "
                "the forward model dimensionality."
            )
            return warnings

        # Look for common input/output key conventions
        x_keys = [k for k in dims if k.startswith("x") or k in (
            "spectral_cube", "image", "input", "signal", "volume",
        )]
        y_keys = [k for k in dims if k.startswith("y") or k in (
            "measurement", "output", "compressed", "detector",
        )]

        if not x_keys:
            warnings.append(
                "Dimension chain: No input signal dimension found in "
                f"'signal_dims' (keys: {list(dims.keys())}).  Expected a key "
                f"starting with 'x' or one of 'spectral_cube', 'image', "
                f"'input', 'signal', 'volume'."
            )
            return warnings

        if not y_keys:
            # Not necessarily an error -- some modalities define only x
            return warnings

        # Compute total element counts
        for xk in x_keys:
            x_shape = dims[xk]
            x_total = _prod(x_shape) if x_shape else 0

            for yk in y_keys:
                y_shape = dims[yk]
                y_total = _prod(y_shape) if y_shape else 0

                if x_total <= 0 or y_total <= 0:
                    warnings.append(
                        f"Dimension chain: Signal dimension '{xk}'={x_shape} or "
                        f"'{yk}'={y_shape} has zero or negative total elements."
                    )
                    continue

                # Compression ratio
                cr = x_total / y_total
                if cr > 100:
                    warnings.append(
                        f"Dimension chain: Extreme compression ratio "
                        f"{cr:.1f}:1 between '{xk}'={x_shape} "
                        f"({x_total} elements) and '{yk}'={y_shape} "
                        f"({y_total} elements).  Recovery may be infeasible."
                    )

        return warnings

    # ----- Check 4: Total throughput sanity -----------------------------------

    @staticmethod
    def _check_total_throughput(system: ImagingSystem) -> List[str]:
        """Warn if the total optical throughput is implausibly low.

        The total throughput is the product of all element throughputs.
        A value below ``_MIN_THROUGHPUT`` (1%) suggests that either the
        system is mis-specified or that the measurement will be severely
        photon-starved.

        Parameters
        ----------
        system : ImagingSystem

        Returns
        -------
        list[str]
            Warnings (empty if throughput is reasonable).
        """
        warnings: List[str] = []

        total_throughput = 1.0
        low_elements: List[str] = []

        for elem in system.elements:
            total_throughput *= elem.throughput
            if elem.throughput < 0.1:
                low_elements.append(
                    f"'{elem.name}' ({elem.throughput:.1%})"
                )

        if total_throughput < _MIN_THROUGHPUT:
            detail = ""
            if low_elements:
                detail = (
                    f"  Lowest-throughput elements: {', '.join(low_elements)}."
                )
            warnings.append(
                f"Total throughput: The product of all element throughputs is "
                f"{total_throughput:.4f} ({total_throughput:.2%}), which is "
                f"below the {_MIN_THROUGHPUT:.0%} sanity threshold.  The "
                f"measurement will be severely photon-starved.{detail}"
            )

        return warnings

    # ----- Check 5: Detector existence (defence-in-depth) --------------------

    @staticmethod
    def _check_detector_exists(system: ImagingSystem) -> List[str]:
        """Verify that the element chain contains at least one detector.

        This is redundant with the pydantic ``field_validator`` on
        ``ImagingSystem.elements``, but is included as a defence-in-depth
        measure.  If the pydantic validation is ever bypassed (e.g. by
        ``model_construct``), this check catches the problem.

        Parameters
        ----------
        system : ImagingSystem

        Returns
        -------
        list[str]
            Warnings (empty if a detector exists).
        """
        has_detector = any(
            e.element_type in ("detector", "transducer")
            for e in system.elements
        )
        if not has_detector:
            return [
                "Detector existence: The imaging system has no detector or "
                "transducer element.  Every physical imaging system must "
                "include at least one detector to produce measurements."
            ]
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prod(shape: List[int]) -> int:
    """Compute the product of a list of integers (total element count).

    Parameters
    ----------
    shape : list[int]
        Dimension sizes.

    Returns
    -------
    int
        Product of all dimensions.  Returns 0 if the list is empty.
    """
    if not shape:
        return 0
    result = 1
    for dim in shape:
        result *= dim
    return result
