"""pwm_core.agents.mismatch_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic Mismatch Agent that quantifies forward-model mismatch.

The agent loads mismatch parameters from the registry YAML for the given
modality, computes a weighted severity score from normalised parameter
errors, and determines an appropriate correction method.  An LLM may
optionally choose the ``mismatch_family_id`` (validated against the
registry); the deterministic fallback uses the registry default.

Design mantra
-------------
The ``run()`` method succeeds without an LLM and produces fully
deterministic outputs.  LLM interaction is limited to soft prior
elicitation (choosing a mismatch family) and is always validated.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAgent, AgentContext
from .contracts import MismatchReport

logger = logging.getLogger(__name__)


class MismatchAgent(BaseAgent):
    """Quantifies forward-model mismatch for a given imaging modality.

    The agent performs three deterministic steps:

    1. **Load** mismatch parameters from the registry (``mismatch_db.yaml``)
       for the modality specified in ``context.modality_key``.
    2. **Compute** a scalar ``severity_score`` as the weighted sum of
       normalised parameter errors:

       .. math::

           S = \\sum_i w_i \\cdot \\frac{|\\text{typical\\_error}_i|}
                                        {|\\text{range\\_width}_i|}

       The score is clipped to [0, 1].
    3. **Estimate** the expected reconstruction improvement (in dB) from
       correcting the mismatch, using the empirical heuristic
       ``improvement = 10 * severity_score`` bounded to [0, 20] dB.

    An optional LLM step asks the model to select a ``mismatch_family_id``
    from the registry.  If the LLM is unavailable or returns an invalid
    key the agent falls back to the registry's ``correction_method``.

    Parameters
    ----------
    llm_client : LLMClient, optional
        If provided, used to select the mismatch family.
    registry : RegistryBuilder, optional
        Source of truth for mismatch parameters.  Required.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, context: AgentContext) -> MismatchReport:
        """Execute the mismatch analysis pipeline.

        Parameters
        ----------
        context : AgentContext
            Must have ``modality_key`` set.  ``imaging_system`` is used
            if available but not required.

        Returns
        -------
        MismatchReport
            Validated pydantic model with severity score, correction
            method, expected improvement, and full parameter details.

        Raises
        ------
        RuntimeError
            If the registry was not provided at construction time.
        KeyError
            If the modality has no entry in ``mismatch_db.yaml``.
        """
        registry = self._require_registry()
        modality_key = context.modality_key

        # --- Step 1: Load mismatch spec from registry -----------------
        mismatch_spec = self._load_mismatch_spec(registry, modality_key)

        parameters = mismatch_spec.parameters
        severity_weights = mismatch_spec.severity_weights
        correction_method = mismatch_spec.correction_method

        # --- Step 2: Build parameter detail dict ----------------------
        param_details = self._build_parameter_details(parameters)

        # --- Step 3: Compute severity score ---------------------------
        severity_score = self._compute_severity_score(
            parameters, severity_weights,
        )

        # --- Step 4: Determine mismatch family (LLM optional) ---------
        mismatch_family = self._select_mismatch_family(
            modality_key, correction_method, registry,
        )

        # --- Step 5: Estimate expected improvement --------------------
        expected_improvement_db = self._compute_expected_improvement(
            severity_score,
        )

        # --- Step 6: Generate explanation -----------------------------
        explanation = self._build_explanation(
            modality_key, severity_score, correction_method,
            expected_improvement_db, param_details,
        )

        # --- Step 7: Collect param_types and subpixel_warnings ----------
        param_types: Dict[str, str] = {}
        subpixel_warnings: List[str] = []
        for name, detail in param_details.items():
            pt = detail.get("param_type", "scale")
            param_types[name] = pt
            fc = detail.get("fidelity_check", {})
            if fc and not fc.get("effective", True):
                warn = fc.get("warning")
                if warn:
                    subpixel_warnings.append(warn)

        return MismatchReport(
            modality_key=modality_key,
            mismatch_family=mismatch_family,
            parameters=param_details,
            severity_score=float(severity_score),
            correction_method=correction_method,
            expected_improvement_db=float(expected_improvement_db),
            explanation=explanation,
            param_types=param_types if param_types else None,
            subpixel_warnings=subpixel_warnings if subpixel_warnings else None,
        )

    # ------------------------------------------------------------------
    # Parameter physics classification
    # ------------------------------------------------------------------

    _VALID_PARAM_TYPES = frozenset({
        "spatial_shift", "rotation", "scale", "blur",
        "offset", "timing", "position",
    })

    _APPLY_METHODS = {
        "spatial_shift": "subpixel_shift_2d",
        "rotation": "subpixel_warp_2d",
        "scale": "scalar_multiply",
        "blur": "gaussian_filter",
        "offset": "scalar_add",
        "timing": "index_permutation",
        "position": "coordinate_perturbation",
    }

    @staticmethod
    def classify_param_physics(name: str, param_spec: Any) -> str:
        """Classify a mismatch parameter by its physics type.

        If ``param_spec.param_type`` is set and valid, return it.
        Otherwise infer from the parameter name and unit.
        """
        pt = getattr(param_spec, "param_type", None)
        if pt is not None and pt in MismatchAgent._VALID_PARAM_TYPES:
            return pt

        name_lower = name.lower()
        unit = getattr(param_spec, "unit", "").lower()

        if any(k in name_lower for k in ("_dx", "_dy", "shift_x", "shift_y",
                                          "detector_offset")) and "pixel" in unit:
            return "spatial_shift"
        if "shift" in name_lower and "pixel" in unit:
            return "spatial_shift"
        if any(k in name_lower for k in ("rotation", "tilt", "angle")) and \
           any(u in unit for u in ("degree", "radian")):
            return "rotation"
        if any(k in name_lower for k in ("gain", "scale")):
            return "scale"
        if "dispersion" in name_lower and "step" in name_lower:
            return "scale"
        if any(k in name_lower for k in ("psf", "blur", "irf_width", "sigma")) and \
           any(u in unit for u in ("pixel", "um")):
            return "blur"
        if any(k in name_lower for k in ("background", "defocus", "bias",
                                          "b0_inhomogeneity")):
            return "offset"
        if any(k in name_lower for k in ("temporal", "timing", "jitter",
                                          "irf_shift")):
            return "timing"
        if any(k in name_lower for k in ("position", "probe")):
            return "position"

        return "scale"

    @staticmethod
    def validate_mismatch_effect(
        name: str, param_type: str, typical_error: float,
    ) -> Dict[str, Any]:
        """Check whether a mismatch parameter produces a measurable effect."""
        warning = None
        effective = True

        if param_type == "spatial_shift" and abs(typical_error) < 0.5:
            effective = False
            warning = (
                f"Parameter '{name}' has typical_error={typical_error} px "
                f"which rounds to 0 with np.roll -- use subpixel_shift_2d."
            )
        elif param_type == "rotation" and abs(typical_error) < 0.01:
            effective = False
            warning = (
                f"Parameter '{name}' has typical_error={typical_error} deg "
                f"which is below measurable threshold."
            )

        return {
            "effective": effective,
            "warning": warning,
            "recommended_method": MismatchAgent._APPLY_METHODS.get(
                param_type, "unknown"
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mismatch_spec(registry: Any, modality_key: str) -> Any:
        """Retrieve the mismatch specification for *modality_key*.

        Parameters
        ----------
        registry : RegistryBuilder
            The loaded registry.
        modality_key : str
            Imaging modality key (e.g. ``"cassi"``).

        Returns
        -------
        MismatchModalityYaml
            Validated mismatch specification from the registry.

        Raises
        ------
        KeyError
            If the modality is not present in the mismatch database.
        """
        # RegistryBuilder stores the mismatch DB as ``_mismatch``
        mismatch_db = registry._mismatch
        modalities = mismatch_db.modalities

        if modality_key not in modalities:
            raise KeyError(
                f"Modality '{modality_key}' not found in mismatch_db.yaml. "
                f"Available modalities: {sorted(modalities.keys())}"
            )
        return modalities[modality_key]

    @staticmethod
    def _build_parameter_details(
        parameters: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Convert registry parameter specs into the report detail format.

        Each entry contains ``typical_error``, ``range``, ``unit``,
        ``description``, and the computed ``normalised_error``.

        Parameters
        ----------
        parameters : dict
            Mapping of parameter name to ``MismatchParamYaml``.

        Returns
        -------
        dict[str, dict[str, Any]]
            Flat dictionary suitable for the ``MismatchReport.parameters``
            field.
        """
        details: Dict[str, Dict[str, Any]] = {}
        for name, param in parameters.items():
            param_range = param.range
            range_width = abs(param_range[1] - param_range[0])

            normalised_error = (
                abs(param.typical_error) / range_width
                if range_width > 1e-12
                else 0.0
            )

            pt = MismatchAgent.classify_param_physics(name, param)
            fidelity = MismatchAgent.validate_mismatch_effect(
                name, pt, param.typical_error,
            )

            details[name] = {
                "typical_error": param.typical_error,
                "range": param_range,
                "range_width": float(range_width),
                "unit": param.unit,
                "description": param.description,
                "normalised_error": float(normalised_error),
                "param_type": pt,
                "apply_method": fidelity["recommended_method"],
                "fidelity_check": fidelity,
            }
        return details

    @staticmethod
    def _compute_severity_score(
        parameters: Dict[str, Any],
        severity_weights: Dict[str, float],
    ) -> float:
        """Compute the aggregate mismatch severity score.

        .. math::

            S = \\text{clip}\\!\\left(
                \\sum_i w_i \\cdot \\frac{|e_i|}{|r_i|}
            ,\\ 0,\\ 1\\right)

        where *w_i* is the weight, *e_i* the typical error, and *r_i*
        the range width for parameter *i*.

        Parameters
        ----------
        parameters : dict
            Mapping of parameter name to ``MismatchParamYaml``.
        severity_weights : dict[str, float]
            Per-parameter weights (should sum to ~1.0).

        Returns
        -------
        float
            Severity score in [0, 1].
        """
        severity = 0.0

        for name, param in parameters.items():
            weight = severity_weights.get(name, 0.0)
            param_range = param.range
            range_width = abs(param_range[1] - param_range[0])

            if range_width < 1e-12:
                # Degenerate range -- skip to avoid division by zero
                logger.warning(
                    "Mismatch parameter '%s' has near-zero range width "
                    "(%.2e); skipping contribution.",
                    name, range_width,
                )
                continue

            normalised_error = abs(param.typical_error) / range_width
            severity += weight * normalised_error

        return float(np.clip(severity, 0.0, 1.0))

    def _select_mismatch_family(
        self,
        modality_key: str,
        default_correction_method: str,
        registry: Any,
    ) -> str:
        """Optionally ask the LLM to choose a mismatch family ID.

        The LLM is presented with the available correction methods and
        modality context.  Its selection is validated against the
        registry.  On any failure the deterministic default
        (``correction_method`` from the registry) is used.

        Parameters
        ----------
        modality_key : str
            Current imaging modality.
        default_correction_method : str
            Fallback family from the registry.
        registry : RegistryBuilder
            Used for validation of the LLM's choice.

        Returns
        -------
        str
            The selected mismatch family identifier.
        """
        if self.llm is None:
            return default_correction_method

        # Collect all valid correction methods across modalities
        valid_families = self._collect_valid_families(registry)

        try:
            prompt = (
                f"For the '{modality_key}' imaging modality, select the "
                f"most appropriate mismatch correction family from the "
                f"available options.  The default is "
                f"'{default_correction_method}'."
            )
            result = self.llm.select(prompt, available_keys=valid_families)
            selected = result.get("selected", default_correction_method)

            # Validate: must be in known families
            if selected in valid_families:
                logger.info(
                    "LLM selected mismatch family '%s' for modality '%s'.",
                    selected, modality_key,
                )
                return selected

            logger.warning(
                "LLM selected unknown mismatch family '%s'; "
                "falling back to registry default '%s'.",
                selected, default_correction_method,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "LLM mismatch family selection failed; "
                "using registry default '%s'.",
                default_correction_method,
                exc_info=True,
            )

        return default_correction_method

    @staticmethod
    def _collect_valid_families(registry: Any) -> List[str]:
        """Gather all unique correction methods across all modalities.

        Parameters
        ----------
        registry : RegistryBuilder
            The loaded registry.

        Returns
        -------
        list[str]
            Sorted list of unique correction method identifiers.
        """
        families: set[str] = set()
        for spec in registry._mismatch.modalities.values():
            families.add(spec.correction_method)
        return sorted(families)

    @staticmethod
    def _compute_expected_improvement(severity_score: float) -> float:
        """Estimate the dB improvement from correcting the mismatch.

        Uses the empirical heuristic:

        .. math::

            \\Delta\\text{PSNR} \\approx 10 \\cdot S

        bounded to [0, 20] dB.

        Parameters
        ----------
        severity_score : float
            Aggregate severity in [0, 1].

        Returns
        -------
        float
            Expected improvement in dB, in [0, 20].
        """
        raw = 10.0 * severity_score
        return float(np.clip(raw, 0.0, 20.0))

    @staticmethod
    def _build_explanation(
        modality_key: str,
        severity_score: float,
        correction_method: str,
        expected_improvement_db: float,
        param_details: Dict[str, Dict[str, Any]],
    ) -> str:
        """Build a deterministic human-readable explanation.

        Parameters
        ----------
        modality_key : str
            Imaging modality.
        severity_score : float
            Computed severity.
        correction_method : str
            Selected correction approach.
        expected_improvement_db : float
            Estimated improvement.
        param_details : dict
            Per-parameter detail dictionaries.

        Returns
        -------
        str
            Multi-line explanation string.
        """
        lines = [
            f"Mismatch analysis for modality '{modality_key}':",
            f"  Severity score: {severity_score:.3f} / 1.000",
            f"  Correction method: {correction_method}",
            f"  Expected improvement: ~{expected_improvement_db:.1f} dB",
            "",
            "  Parameter contributions:",
        ]

        # Sort parameters by normalised error (descending) for readability
        sorted_params = sorted(
            param_details.items(),
            key=lambda kv: kv[1].get("normalised_error", 0.0),
            reverse=True,
        )
        for name, detail in sorted_params:
            ne = detail.get("normalised_error", 0.0)
            te = detail.get("typical_error", 0.0)
            unit = detail.get("unit", "")
            pt = detail.get("param_type", "")
            lines.append(
                f"    {name}: typical_error={te} {unit}, "
                f"normalised={ne:.3f} [{pt}]"
            )

        # Fidelity warnings
        fidelity_warnings = []
        for name, detail in param_details.items():
            fc = detail.get("fidelity_check", {})
            if fc and not fc.get("effective", True):
                warn = fc.get("warning", "")
                if warn:
                    fidelity_warnings.append(warn)
        if fidelity_warnings:
            lines.append("")
            lines.append("  Fidelity warnings:")
            for w in fidelity_warnings:
                lines.append(f"    - {w}")

        return "\n".join(lines)
