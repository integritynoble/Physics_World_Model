"""pwm_core.core.runbundle.triad_report

Domain-diagnostic report hierarchy for RunBundle artifacts.

DomainDiagnosticReport (base)
-----------------------------
Abstract base model for domain-specific diagnostic reports.  Each domain
subclasses this with its own diagnostic fields.  Reports are stored at
``<bundle_dir>/logs/<domain>_diagnostic_report.json``.

ImagingTriadReport (imaging domain)
------------------------------------
Detailed G1/G2/G3 gate attribution artifact for inverse-imaging modalities.
Elaborates the brief :class:`ImagingTriadFlags` summary stored in
``certificate.json`` with full evidence dicts, per-scenario PSNR breakdowns,
confidence scores, dominant-gate identification, and methodology identifiers.

Relationship to ImagingTriadFlags in certificate.json
------------------------------------------------------
* **ImagingTriadFlags** (in ``certificate.json``) is the *summary*: three
  verdict fields (pass/warn/fail) included directly in the Certificate object
  for fast inspection at the bundle root.
* **ImagingTriadReport** (stored at ``logs/imaging_diagnostic_report.json``)
  is the *detailed* artifact: full evidence dicts, per-scenario PSNR
  breakdowns, confidence scores, dominant-gate identification, and the
  methodology identifier.

Use ImagingTriadFlags for CI badge rendering and quick trust-tier decisions.
Use ImagingTriadReport for post-hoc root-cause analysis and scientific audits.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    from dataclasses import dataclass as BaseModel  # type: ignore
    Field = lambda *a, **kw: None  # type: ignore


# ---------------------------------------------------------------------------
# Sub-model
# ---------------------------------------------------------------------------

class GateAttribution(BaseModel):
    """Detailed attribution for one triad gate (G1, G2, or G3).

    ``confidence`` is a float in [0, 1] indicating how strongly the evidence
    points to this gate as a bottleneck.  ``verdict`` summarises the finding
    using the same pass/warn/fail vocabulary as the R-gate results in
    ``certificate.json``.  ``evidence`` is a gate-specific dict whose schema
    varies by gate type -- common keys include ``oracle_gap_db``,
    ``noise_snr_db``, and ``operator_mismatch_ratio``.
    """

    gate: str = Field(
        ...,
        description="Gate identifier: 'g1', 'g2', or 'g3'",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Attribution confidence in [0, 1]",
    )
    verdict: str = Field(
        ...,
        description="Gate verdict: 'pass', 'warn', or 'fail'",
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Gate-specific evidence dictionary for audit and debugging",
    )


# ---------------------------------------------------------------------------
# DomainDiagnosticReport (base)
# ---------------------------------------------------------------------------

class DomainDiagnosticReport(BaseModel):
    """Base class for domain-specific diagnostic reports.

    Each domain subclasses this with its own diagnostic fields.
    Stored at ``<bundle_dir>/logs/<domain>_diagnostic_report.json``.
    """

    run_id: str = Field(..., description="Run identifier matching the RunBundle")
    spec_id: str = Field(..., description="CoreSpec ID that governed this run")
    domain: str = Field(..., description="Domain identifier, e.g. 'imaging', 'ct_qc', 'combustion'")
    issued_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when this report was issued",
    )
    methodology: str = Field(
        default="unspecified",
        description="Identifier of the diagnostic methodology used to produce this report",
    )

    @property
    def is_complete(self) -> bool:
        """Whether all required diagnostics have been evaluated.

        Subclasses must override this property with domain-specific logic.
        """
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        try:
            return self.model_dump()
        except AttributeError:
            return self.__dict__


# ---------------------------------------------------------------------------
# ImagingTriadReport
# ---------------------------------------------------------------------------

class ImagingTriadReport(DomainDiagnosticReport):
    """Detailed G1/G2/G3 gate attribution artifact for imaging modalities.

    ImagingTriadReport elaborates the triad attribution far beyond the brief
    :class:`ImagingTriadFlags` summary stored in ``certificate.json``.  It is
    written to ``logs/imaging_diagnostic_report.json`` inside the RunBundle
    directory.

    Gate definitions
    ----------------
    * **G1 (sampling/forward model)** -- mismatch between the assumed and actual
      measurement operator; dominated by oracle-gap metrics across DR-IS
      scenarios I-IV.
    * **G2 (noise model)** -- mismatch between the assumed noise distribution
      and actual noise; dominated by scenario II vs. scenario I residuals.
    * **G3 (operator mismatch)** -- structural operator error (e.g. wrong PSF,
      wrong discretisation); dominated by scenario III vs. scenario I
      residuals.

    See Also
    --------
    ImagingTriadFlags : summary triad verdict stored in certificate.json
    """

    # Domain default
    domain: str = Field(default="imaging", description="Domain identifier")

    # Gate attributions
    g1_sampling: GateAttribution = Field(
        ...,
        description="Detailed G1 (sampling/forward model) attribution",
    )
    g2_noise: GateAttribution = Field(
        ...,
        description="Detailed G2 (noise model) attribution",
    )
    g3_operator: GateAttribution = Field(
        ...,
        description="Detailed G3 (operator mismatch) attribution",
    )

    # Dominant gate summary
    dominant_gate: Optional[str] = Field(
        default=None,
        description="Gate with highest confidence: 'g1', 'g2', 'g3', or None if ambiguous",
    )
    dominant_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence of the dominant gate determination",
    )

    # Per-scenario metrics
    psnr_by_scenario: Dict[str, float] = Field(
        default_factory=dict,
        description="PSNR (dB) keyed by scenario_id, e.g. {'I': 32.1, 'II': 29.4, ...}",
    )

    # Methodology tag
    methodology: str = Field(
        default="infer_gate_attribution_v1",
        description="Identifier of the attribution methodology used to produce this report",
    )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True if all three gates have a positive confidence estimate.

        A report is considered complete when the attribution algorithm has
        produced a non-zero confidence for every gate, meaning that evidence
        was actually collected for G1, G2, and G3.  A report with any zero
        confidence indicates that at least one gate was not evaluated.
        """
        return (
            self.g1_sampling.confidence > 0.0
            and self.g2_noise.confidence > 0.0
            and self.g3_operator.confidence > 0.0
        )


# Backward compatibility alias -- existing code imports TriadReport
TriadReport = ImagingTriadReport
