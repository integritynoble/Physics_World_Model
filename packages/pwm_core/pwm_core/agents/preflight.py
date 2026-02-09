"""
pwm_core.agents.preflight
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-flight report builder and CLI permit logic.

This module assembles the :class:`PreFlightReport` from all sub-agent
outputs, estimates runtime, and gates execution through one of three
permit modes (``interactive``, ``auto_proceed``, ``force``).

Design invariants
-----------------
* The report builder is deterministic -- no LLM calls.
* ``check_permit`` never blocks in non-interactive environments; it falls
  back to ``proceed_recommended``.
* ``format_preflight_report`` produces a human-readable ASCII box suitable
  for terminal display or log capture.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from functools import reduce
from typing import Dict, List, Optional

from .contracts import (
    ImagingSystem,
    MismatchReport,
    ModalitySelection,
    PhotonReport,
    PreFlightReport,
    RecoverabilityReport,
    SystemAnalysis,
    NegotiationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permit modes
# ---------------------------------------------------------------------------


class PermitMode(str, Enum):
    """How the CLI should gate execution after the pre-flight report."""

    interactive = "interactive"
    auto_proceed = "auto_proceed"
    force = "force"


# ---------------------------------------------------------------------------
# Solver complexity heuristics
# ---------------------------------------------------------------------------

# Rough FLOP multipliers per solver family, used for runtime estimation.
# These are *relative* values -- the actual runtime is proportional to
# system dimensions x complexity multiplier.
_SOLVER_COMPLEXITY: Dict[str, float] = {
    # Traditional CPU solvers
    "gap_tv": 1.0,
    "richardson_lucy": 0.8,
    "fbp": 0.3,
    "sense": 1.2,
    "epie": 1.5,
    "angular_spectrum": 0.4,
    "admm_tv": 1.2,
    "tval3": 1.0,
    "wiener_sim": 0.5,
    "fista_l2": 0.8,
    "fourier_notch": 0.3,
    "laplacian_fusion": 0.3,
    # Deep learning solvers
    "mst": 2.5,
    "hdnet": 3.0,
    "care": 2.0,
    "noise2void": 1.8,
    "dl_sim": 2.5,
    "efficientsci": 3.5,
    "elp_unfolding": 3.0,
    "flatnet": 4.0,
    "destripe": 2.0,
    "redcnn": 2.0,
    "varnet": 3.5,
    "modl": 2.5,
    "ptychonn": 2.5,
    "phasenet": 2.0,
    "nerf": 10.0,
    "instant_ngp": 5.0,
    "gaussian_splatting": 8.0,
    "lista": 1.5,
    "ifcnn": 1.5,
    "diffusion_posterior": 15.0,
    "ista_net": 2.0,
    "hatnet": 3.0,
    "hsi_sdecnn": 1.5,
}

# Upload file templates per modality for operator correction fallback.
# These describe what the user needs to upload for Mode 2 (operator
# correction / UPWMI) when the ideal forward model does not match the
# real measurement.
_UPLOAD_TEMPLATES: Dict[str, List[str]] = {
    "cassi": [
        "y.npy  -- measured 2D snapshot (H+S-1, W)",
        "mask.npy  -- coded aperture pattern (H, W) binary",
        "dispersion_step (float)  -- pixels/channel",
    ],
    "spc": [
        "y.npy  -- compressed measurements (M,)",
        "Phi.npy  -- measurement matrix (M, N)",
    ],
    "cacti": [
        "y.npy  -- temporal snapshot (H, W)",
        "masks.npy  -- temporal mask stack (T, H, W)",
    ],
    "widefield": [
        "y.npy  -- raw image (H, W)",
        "psf.npy  -- measured PSF kernel (kH, kW)",
    ],
    "widefield_lowdose": [
        "y.npy  -- raw low-dose image (H, W)",
        "psf.npy  -- measured PSF kernel (kH, kW)",
    ],
    "confocal_livecell": [
        "y.npy  -- raw confocal frame (H, W)",
        "psf.npy  -- measured confocal PSF (kH, kW)",
    ],
    "confocal_3d": [
        "y.npy  -- raw 3D confocal stack (D, H, W)",
        "psf.npy  -- 3D PSF kernel (kD, kH, kW)",
    ],
    "sim": [
        "y_frames.npy  -- raw SIM frames (N_orientations * N_phases, H, W)",
        "illumination_params.json  -- pattern angles, frequencies, phases",
    ],
    "lensless": [
        "y.npy  -- raw sensor measurement (H, W)",
        "psf.npy  -- calibrated point-spread function (H, W)",
    ],
    "lightsheet": [
        "y.npy  -- raw light-sheet image (H, W) or (D, H, W)",
    ],
    "ct": [
        "sinogram.npy  -- projection data (N_angles, N_detectors)",
        "angles.npy  -- projection angles in radians (N_angles,)",
    ],
    "mri": [
        "kspace.npy  -- undersampled k-space (N_coils, kH, kW) complex",
        "mask.npy  -- sampling mask (kH, kW) binary",
        "sensitivity_maps.npy  -- coil sensitivities (N_coils, H, W) complex",
    ],
    "ptychography": [
        "diffraction_patterns.npy  -- stack of patterns (N_positions, pH, pW)",
        "positions.npy  -- scan positions (N_positions, 2)",
        "probe_guess.npy  -- initial probe estimate (pH, pW) complex",
    ],
    "holography": [
        "hologram.npy  -- raw hologram (H, W)",
        "wavelength_nm (float)  -- illumination wavelength",
        "pixel_pitch_um (float)  -- detector pixel size",
        "propagation_distance_um (float)  -- sample-to-sensor distance",
    ],
    "nerf": [
        "images/  -- directory of posed images",
        "transforms.json  -- camera intrinsics and extrinsics per frame",
    ],
    "gaussian_splatting": [
        "images/  -- directory of posed images",
        "transforms.json  -- camera intrinsics and extrinsics per frame",
        "points3D.ply  -- initial SfM point cloud (optional)",
    ],
    "matrix": [
        "y.npy  -- measurement vector (M,)",
        "A.npy  -- forward matrix (M, N)",
    ],
    "panorama": [
        "images/  -- directory of overlapping images",
        "focaldepths.json  -- per-image focus distances (optional)",
    ],
    "panorama_multifocal": [
        "images/  -- directory of multi-focus images",
        "focaldepths.json  -- per-image focus distances",
    ],
}


# ---------------------------------------------------------------------------
# PreFlightReportBuilder
# ---------------------------------------------------------------------------


class PreFlightReportBuilder:
    """Assembles a :class:`PreFlightReport` from all sub-agent outputs.

    The builder is stateless and deterministic. It combines negotiation
    outcomes, runtime estimates, and upload guidance into a single report
    suitable for user review.
    """

    def build(
        self,
        modality: ModalitySelection,
        system: ImagingSystem,
        photon: PhotonReport,
        mismatch: MismatchReport,
        recoverability: RecoverabilityReport,
        analysis: SystemAnalysis,
        negotiation: NegotiationResult,
    ) -> PreFlightReport:
        """Build the complete pre-flight report.

        Parameters
        ----------
        modality : ModalitySelection
            Which modality was selected and why.
        system : ImagingSystem
            The constructed imaging system with element chain.
        photon : PhotonReport
            Photon budget analysis.
        mismatch : MismatchReport
            Operator mismatch analysis.
        recoverability : RecoverabilityReport
            Recoverability and compression analysis.
        analysis : SystemAnalysis
            Bottleneck diagnosis and suggestions.
        negotiation : NegotiationResult
            Outcome of inter-agent negotiation.

        Returns
        -------
        PreFlightReport
            Complete report ready for user review and permit check.
        """
        # -- Collect warnings from negotiation vetoes ----------------------
        warnings: List[str] = []
        for veto in negotiation.vetoes:
            warning_msg = f"[{veto.source}] {veto.reason}"
            if veto.suggested_resolution:
                resolutions = "; ".join(veto.suggested_resolution)
                warning_msg += f" (suggestions: {resolutions})"
            warnings.append(warning_msg)

        # Add quality-tier warnings
        if photon.quality_tier == "insufficient":
            warnings.append(
                f"Photon budget is insufficient: SNR = {photon.snr_db:.1f} dB, "
                f"regime = {photon.noise_regime.value}"
            )
        elif photon.quality_tier == "marginal":
            warnings.append(
                f"Photon budget is marginal: SNR = {photon.snr_db:.1f} dB"
            )

        if mismatch.severity_score > 0.7:
            warnings.append(
                f"High operator mismatch severity: {mismatch.severity_score:.2f} "
                f"({mismatch.mismatch_family})"
            )

        if recoverability.verdict == "insufficient":
            warnings.append(
                f"Recoverability is insufficient: score = "
                f"{recoverability.recoverability_score:.2f}, "
                f"expected PSNR = {recoverability.expected_psnr_db:.1f} dB"
            )

        # -- Estimate runtime ----------------------------------------------
        estimated_runtime_s = self._estimate_runtime(system, recoverability)

        # -- Determine what to upload for operator correction fallback -----
        what_to_upload = self._get_upload_template(modality.modality_key)

        # -- Set proceed recommendation from negotiation -------------------
        proceed_recommended = negotiation.proceed

        return PreFlightReport(
            modality=modality,
            system=system,
            photon=photon,
            mismatch=mismatch,
            recoverability=recoverability,
            analysis=analysis,
            estimated_runtime_s=estimated_runtime_s,
            proceed_recommended=proceed_recommended,
            warnings=warnings,
            what_to_upload=what_to_upload,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_runtime(
        self,
        system: ImagingSystem,
        recoverability: RecoverabilityReport,
    ) -> float:
        """Estimate wall-clock runtime in seconds.

        The estimate is based on:
        1. Total pixel count from signal dimensions.
        2. Solver complexity multiplier from the recommended solver family.
        3. Compression ratio (higher CR -> more iterations typically).

        This is a rough heuristic, not a benchmark. Intended for user
        expectation management, not scheduling.
        """
        # Compute total dimensionality (product of all signal dims)
        total_pixels = 1
        for dim_name, dim_shape in system.signal_dims.items():
            total_pixels *= reduce(lambda a, b: a * b, dim_shape, 1)

        # Normalise to a "256x256 image equivalent" baseline
        baseline_pixels = 256 * 256
        dim_factor = max(total_pixels / baseline_pixels, 0.1)

        # Look up solver complexity
        solver_key = recoverability.recommended_solver_family.lower()
        # Try exact match first, then prefix match
        complexity = _SOLVER_COMPLEXITY.get(solver_key, None)
        if complexity is None:
            for known_key, known_complexity in _SOLVER_COMPLEXITY.items():
                if known_key in solver_key or solver_key in known_key:
                    complexity = known_complexity
                    break
        if complexity is None:
            complexity = 2.0  # default for unknown solvers

        # Compression ratio scaling: higher CR -> ~linear increase
        cr_factor = max(recoverability.compression_ratio, 1.0) / 8.0
        cr_factor = max(cr_factor, 0.5)

        # Base time for a 256x256 baseline operation: ~2 seconds
        base_time_s = 2.0
        estimated_s = base_time_s * dim_factor * complexity * cr_factor

        # Clamp to reasonable range [0.1, 7200] seconds
        return max(0.1, min(estimated_s, 7200.0))

    def _get_upload_template(
        self, modality_key: str
    ) -> Optional[List[str]]:
        """Return the upload guidance list for a given modality.

        If the modality is not in the template table, returns a generic
        fallback guidance list.
        """
        template = _UPLOAD_TEMPLATES.get(modality_key)
        if template is not None:
            return template

        # Generic fallback
        return [
            "y.npy  -- measured data",
            "A.npy  -- forward operator (if applicable)",
            "metadata.json  -- acquisition parameters",
        ]


# ---------------------------------------------------------------------------
# Permit check
# ---------------------------------------------------------------------------


def check_permit(report: PreFlightReport, mode: PermitMode) -> bool:
    """Gate execution based on the pre-flight report and permit mode.

    Parameters
    ----------
    report : PreFlightReport
        The assembled pre-flight report.
    mode : PermitMode
        How to handle the go/no-go decision.

    Returns
    -------
    bool
        True if execution should proceed, False otherwise.

    Behaviour per mode
    ------------------
    ``interactive``
        Prints the formatted report. In a genuine interactive terminal,
        returns True (user is expected to read and interrupt if needed).
        In non-interactive environments (pipes, CI), logs the report and
        returns ``proceed_recommended``.

    ``auto_proceed``
        Returns ``proceed_recommended`` directly. Logs a warning if the
        recommendation is to *not* proceed.

    ``force``
        Always returns True. Logs a warning that safety checks are being
        overridden.
    """
    if mode == PermitMode.interactive:
        formatted = format_preflight_report(report)

        # Check if we are in an interactive terminal
        is_interactive = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()

        if is_interactive:
            print(formatted)
            logger.info("Pre-flight report displayed to user (interactive).")
            return True
        else:
            # Non-interactive: log and use recommendation
            logger.info(
                "Non-interactive environment detected. Pre-flight report:\n%s",
                formatted,
            )
            if not report.proceed_recommended:
                logger.warning(
                    "Pre-flight report recommends NOT proceeding. "
                    "Returning proceed_recommended=False."
                )
            return report.proceed_recommended

    elif mode == PermitMode.auto_proceed:
        if not report.proceed_recommended:
            logger.warning(
                "Pre-flight check: proceed_recommended is False. "
                "Warnings: %s",
                "; ".join(report.warnings) if report.warnings else "(none)",
            )
        else:
            logger.info(
                "Pre-flight check: auto-proceeding (proceed_recommended=True)."
            )
        return report.proceed_recommended

    elif mode == PermitMode.force:
        if not report.proceed_recommended:
            logger.warning(
                "FORCE mode: overriding negative pre-flight recommendation. "
                "Warnings: %s",
                "; ".join(report.warnings) if report.warnings else "(none)",
            )
        else:
            logger.warning(
                "FORCE mode: proceeding regardless of pre-flight checks."
            )
        return True

    else:
        # Should not happen with the enum, but be defensive.
        logger.error("Unknown permit mode: %s. Defaulting to False.", mode)
        return False


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------


def format_preflight_report(report: PreFlightReport) -> str:
    """Format a :class:`PreFlightReport` as a human-readable ASCII box.

    Parameters
    ----------
    report : PreFlightReport
        The pre-flight report to format.

    Returns
    -------
    str
        Multi-line ASCII-formatted string suitable for terminal display.
    """
    width = 72
    border = "+" + "-" * (width - 2) + "+"
    title_line = _center_text("PRE-FLIGHT REPORT", width)

    lines: List[str] = []
    lines.append(border)
    lines.append(title_line)
    lines.append(border)

    # -- Modality ----------------------------------------------------------
    lines.append(_kv_line("Modality", report.modality.modality_key, width))
    lines.append(
        _kv_line("Confidence", f"{report.modality.confidence:.0%}", width)
    )
    lines.append(
        _kv_line("Reasoning", report.modality.reasoning[:50], width)
    )

    lines.append(_separator(width))

    # -- Photon budget -----------------------------------------------------
    lines.append(
        _kv_line("Photon SNR", f"{report.photon.snr_db:.1f} dB", width)
    )
    lines.append(
        _kv_line("Noise regime", report.photon.noise_regime.value, width)
    )
    lines.append(
        _kv_line("Quality tier", report.photon.quality_tier, width)
    )

    lines.append(_separator(width))

    # -- Mismatch ----------------------------------------------------------
    lines.append(
        _kv_line(
            "Mismatch severity",
            f"{report.mismatch.severity_score:.2f}",
            width,
        )
    )
    lines.append(
        _kv_line(
            "Mismatch family", report.mismatch.mismatch_family, width
        )
    )
    lines.append(
        _kv_line(
            "Correction method", report.mismatch.correction_method, width
        )
    )

    lines.append(_separator(width))

    # -- Recoverability ----------------------------------------------------
    lines.append(
        _kv_line(
            "Recoverability",
            f"{report.recoverability.recoverability_score:.2f} "
            f"(conf: {report.recoverability.recoverability_confidence:.2f})",
            width,
        )
    )
    lines.append(
        _kv_line(
            "Expected PSNR",
            f"{report.recoverability.expected_psnr_db:.1f} "
            f"+/- {report.recoverability.expected_psnr_uncertainty_db:.1f} dB",
            width,
        )
    )
    lines.append(
        _kv_line("Verdict", report.recoverability.verdict, width)
    )

    lines.append(_separator(width))

    # -- Analysis bottleneck -----------------------------------------------
    lines.append(
        _kv_line(
            "Primary bottleneck",
            report.analysis.primary_bottleneck,
            width,
        )
    )
    lines.append(
        _kv_line(
            "P(success)",
            f"{report.analysis.probability_of_success:.0%}",
            width,
        )
    )

    lines.append(_separator(width))

    # -- Runtime estimate --------------------------------------------------
    runtime = report.estimated_runtime_s
    if runtime < 60:
        runtime_str = f"{runtime:.1f} s"
    elif runtime < 3600:
        runtime_str = f"{runtime / 60:.1f} min"
    else:
        runtime_str = f"{runtime / 3600:.1f} h"
    lines.append(_kv_line("Est. runtime", runtime_str, width))

    # -- Proceed recommendation --------------------------------------------
    proceed_str = "YES" if report.proceed_recommended else "NO"
    lines.append(
        _kv_line("Proceed recommended", proceed_str, width)
    )

    # -- Warnings ----------------------------------------------------------
    if report.warnings:
        lines.append(_separator(width))
        lines.append(_center_text("WARNINGS", width))
        for i, warning in enumerate(report.warnings, 1):
            # Wrap long warnings
            prefix = f"  {i}. "
            max_warn_len = width - len(prefix) - 4
            if len(warning) <= max_warn_len:
                lines.append(_pad_line(f"{prefix}{warning}", width))
            else:
                # Simple word-wrap
                words = warning.split()
                current_line = prefix
                for word in words:
                    if len(current_line) + len(word) + 1 <= width - 4:
                        current_line += word + " "
                    else:
                        lines.append(_pad_line(current_line.rstrip(), width))
                        current_line = "     " + word + " "
                if current_line.strip():
                    lines.append(_pad_line(current_line.rstrip(), width))

    lines.append(border)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _center_text(text: str, width: int) -> str:
    """Center text within a bordered line."""
    inner = width - 4  # account for "| " and " |"
    centered = text.center(inner)
    return f"| {centered} |"


def _kv_line(key: str, value: str, width: int) -> str:
    """Format a key-value pair within a bordered line."""
    inner = width - 4
    kv = f"{key}: {value}"
    if len(kv) > inner:
        kv = kv[:inner]
    padded = kv.ljust(inner)
    return f"| {padded} |"


def _pad_line(text: str, width: int) -> str:
    """Pad text to fit within a bordered line."""
    inner = width - 4
    if len(text) > inner:
        text = text[:inner]
    padded = text.ljust(inner)
    return f"| {padded} |"


def _separator(width: int) -> str:
    """A thin separator line within the box."""
    return "|" + "." * (width - 2) + "|"
