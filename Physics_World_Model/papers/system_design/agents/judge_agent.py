"""Judge Agent — validates designs via the Triad diagnostic gates.

Three-stage validation:
  Stage 1 (Prerequisite): 6-gate structural compiler
    - DAG acyclicity, canonical chain, complexity bounds,
      nonlinear constraints, adjoint consistency, representation error

  Stage 2 (Primary): Triad gate evaluation (from flagship paper)
    - Gate 1 (Recoverability): compression ratio → information capacity
    - Gate 2 (Carrier Budget): SNR from source + detector + noise
    - Gate 3 (Operator Mismatch): calibration tolerances → deployment PSNR

  Stage 3 (System): Cost estimation + physical feasibility
    - Itemized hardware cost from system_elements
    - Physical realizability of each element
"""
from __future__ import annotations

import math
from typing import Literal, Any

from ..config import JUDGE_MODEL
from ..schemas import PlanDocument, ForwardAction, ReconAction
from ..schemas.judgment import (
    JudgmentResult,
    JudgmentIssue,
    TriadGateResult,
    TriadReport,
    CostItem,
    CostEstimate,
    FeasibilityItem,
    FeasibilityReport,
)
from .base_agent import BaseAgent


# ── Triad Gate Evaluator (deterministic) ──────────────────────────────────────

class TriadGateEvaluator:
    """Deterministic evaluation of the three Triad gates from the flagship paper.

    Each gate produces a PASS/FAIL verdict with a quantitative margin in dB.
    Gate scoring functions are taken directly from the Compression Bound,
    Noise Bound, and Calibration Sensitivity theorems.
    """

    # Modality-specific compression thresholds (gamma → PSNR_max_G1)
    # Lookup table from flagship paper's compression_db
    _COMPRESSION_DB: dict[str, dict[str, float]] = {
        # modality: {gamma_threshold: psnr_at_threshold, ...}
        "computed_tomography": {"min_gamma": 0.15, "psnr_at_min": 18.0, "slope": 15.0},
        "ct": {"min_gamma": 0.15, "psnr_at_min": 18.0, "slope": 15.0},
        "mri": {"min_gamma": 0.20, "psnr_at_min": 20.0, "slope": 18.0},
        "cassi": {"min_gamma": 0.03, "psnr_at_min": 22.0, "slope": 12.0},
        "cacti": {"min_gamma": 0.10, "psnr_at_min": 25.0, "slope": 14.0},
        "lensless": {"min_gamma": 0.80, "psnr_at_min": 6.0, "slope": 8.0},
        "sim": {"min_gamma": 0.50, "psnr_at_min": 24.0, "slope": 10.0},
        "ultrasound": {"min_gamma": 0.10, "psnr_at_min": 22.0, "slope": 16.0},
        "oct": {"min_gamma": 0.30, "psnr_at_min": 25.0, "slope": 14.0},
        "ptychography": {"min_gamma": 0.40, "psnr_at_min": 24.0, "slope": 12.0},
        "dot": {"min_gamma": 0.20, "psnr_at_min": 15.0, "slope": 10.0},
    }
    # Default for unknown modalities
    _DEFAULT_COMPRESSION = {"min_gamma": 0.10, "psnr_at_min": 15.0, "slope": 12.0}

    @classmethod
    def evaluate_gate1(
        cls,
        plan: PlanDocument,
        system_elements: dict[str, Any] | None = None,
    ) -> TriadGateResult:
        """Gate 1 (Recoverability): check compression ratio against information capacity.

        PSNR_max_G1 = f(modality, gamma) where gamma = m/n.
        """
        modality = plan.modality.lower().replace(" ", "_")
        params = cls._COMPRESSION_DB.get(modality, cls._DEFAULT_COMPRESSION)

        # Extract measurement and signal dimensions from the plan
        m, n = cls._extract_dimensions(plan, system_elements)
        gamma = m / n if n > 0 else 0.0

        # Compression bound: PSNR_max_G1 = psnr_at_min + slope * (gamma - min_gamma)
        if gamma < params["min_gamma"]:
            psnr_max_g1 = params["psnr_at_min"] * (gamma / params["min_gamma"])
        else:
            psnr_max_g1 = params["psnr_at_min"] + params["slope"] * (gamma - params["min_gamma"])

        # Get target PSNR
        target_psnr = cls._extract_target_psnr(plan)

        passed = psnr_max_g1 >= target_psnr
        margin = psnr_max_g1 - target_psnr

        return TriadGateResult(
            gate="G1_recoverability",
            passed=passed,
            margin_db=round(margin, 1),
            score_db=round(psnr_max_g1, 1),
            detail=(
                f"Compression ratio gamma={gamma:.3f} (m={m}, n={n}). "
                f"PSNR_max_G1={psnr_max_g1:.1f} dB vs target={target_psnr:.1f} dB. "
                f"{'PASS' if passed else 'FAIL'} (margin={margin:+.1f} dB)."
            ),
            design_action="" if passed else (
                f"Information deficiency: gamma={gamma:.3f} insufficient for "
                f"target {target_psnr:.1f} dB. Increase sampling density or add views."
            ),
        )

    @classmethod
    def evaluate_gate2(
        cls,
        plan: PlanDocument,
        system_elements: dict[str, Any] | None = None,
    ) -> TriadGateResult:
        """Gate 2 (Carrier Budget): check measurement SNR against noise floor.

        SNR = QE * N_photon / sqrt(QE * N_photon + sigma_read^2 + I_dark * t_exp)
        PSNR_max_G2 = 10*log10(SNR) + C_M
        """
        # Extract noise and detector parameters
        snr_db = cls._compute_measurement_snr(plan, system_elements)
        target_psnr = cls._extract_target_psnr(plan)

        # Conservative C_M estimate (depends on conditioning)
        c_m = cls._estimate_conditioning_factor(plan)
        psnr_max_g2 = snr_db + c_m

        passed = psnr_max_g2 >= target_psnr
        margin = psnr_max_g2 - target_psnr

        return TriadGateResult(
            gate="G2_carrier_budget",
            passed=passed,
            margin_db=round(margin, 1),
            score_db=round(psnr_max_g2, 1),
            detail=(
                f"Measurement SNR={snr_db:.1f} dB, conditioning factor C_M={c_m:.1f} dB. "
                f"PSNR_max_G2={psnr_max_g2:.1f} dB vs target={target_psnr:.1f} dB. "
                f"{'PASS' if passed else 'FAIL'} (margin={margin:+.1f} dB)."
            ),
            design_action="" if passed else (
                f"Carrier budget insufficient: SNR={snr_db:.1f} dB, noise floor exceeds target "
                f"by {abs(margin):.1f} dB. Increase source power, exposure time, or detector QE."
            ),
        )

    @classmethod
    def evaluate_gate3(
        cls,
        plan: PlanDocument,
        system_elements: dict[str, Any] | None = None,
    ) -> TriadGateResult:
        """Gate 3 (Operator Mismatch): check forward model fidelity.

        Delta_PSNR_total = sqrt(sum_k (sensitivity_k * tolerance_k)^2)
        PSNR_deploy = PSNR_max_G2 - Delta_PSNR_total
        """
        # Get Gate 2 score as baseline
        g2 = cls.evaluate_gate2(plan, system_elements)
        psnr_max_g2 = g2.score_db

        # Compute mismatch degradation from calibration tolerances
        delta_psnr_total, dominant_param, details = cls._compute_mismatch_degradation(
            plan, system_elements
        )

        # Recovery ratio (from autonomous calibration)
        recovery_ratio = cls._extract_recovery_ratio(system_elements)
        effective_degradation = delta_psnr_total * (1.0 - recovery_ratio)

        psnr_deploy = psnr_max_g2 - effective_degradation
        target_psnr = cls._extract_target_psnr(plan)

        passed = psnr_deploy >= target_psnr
        margin = psnr_deploy - target_psnr

        return TriadGateResult(
            gate="G3_operator_mismatch",
            passed=passed,
            margin_db=round(margin, 1),
            score_db=round(psnr_deploy, 1),
            detail=(
                f"Expected mismatch degradation: {delta_psnr_total:.1f} dB "
                f"(dominant: {dominant_param}). "
                f"Recovery ratio: {recovery_ratio:.0%} -> effective loss: "
                f"{effective_degradation:.1f} dB. "
                f"PSNR_deploy={psnr_deploy:.1f} dB vs target={target_psnr:.1f} dB. "
                f"{'PASS' if passed else 'FAIL'} (margin={margin:+.1f} dB). "
                f"Calibration details: {details}"
            ),
            design_action="" if passed else (
                f"Gate 3 risk: deployment degradation={delta_psnr_total:.1f} dB. "
                f"Dominant parameter: {dominant_param}. "
                f"Action: tighten {dominant_param} tolerance or increase calibration frequency."
            ),
        )

    @classmethod
    def evaluate_all(
        cls,
        plan: PlanDocument,
        system_elements: dict[str, Any] | None = None,
    ) -> TriadReport:
        """Run all three Triad gates and return a complete report."""
        g1 = cls.evaluate_gate1(plan, system_elements)
        g2 = cls.evaluate_gate2(plan, system_elements)
        g3 = cls.evaluate_gate3(plan, system_elements)

        # Determine dominant gate (largest potential MSE contribution)
        target = cls._extract_target_psnr(plan)
        deficits = {
            "G1": max(0, target - g1.score_db),
            "G2": max(0, target - g2.score_db),
            "G3": max(0, target - g3.score_db),
        }
        dominant = max(deficits, key=deficits.get)  # type: ignore[arg-type]

        triad_passed = g1.passed and g2.passed and g3.passed

        return TriadReport(
            gate1=g1,
            gate2=g2,
            gate3=g3,
            dominant_gate=dominant,
            triad_passed=triad_passed,
            expected_deployment_psnr_db=g3.score_db if g3.score_db > 0 else None,
            recovery_ratio=cls._extract_recovery_ratio(system_elements),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_dimensions(
        plan: PlanDocument,
        system_elements: dict[str, Any] | None,
    ) -> tuple[int, int]:
        """Extract measurement (m) and signal (n) dimensions from the plan."""
        # Try to parse from action
        if isinstance(plan.action, ForwardAction):
            # Look for measurement_shape and object dimensions in elements
            m, n = 0, 0
            for el in plan.action.elements:
                params = el.parameters
                if el.type == "detector":
                    pixels = params.get("pixels", params.get("resolution", [256, 256]))
                    if isinstance(pixels, list) and len(pixels) >= 2:
                        m = max(m, pixels[0] * pixels[1])
                if "n_angles" in params:
                    n_angles = params["n_angles"]
                    n_det = params.get("n_detectors", params.get("n_det", 256))
                    m = n_angles * n_det
            # Default estimates if not found
            if m == 0:
                m = 256 * 256
            if n == 0:
                n = 256 * 256
            return m, n
        # For reconstruction, use defaults
        return 256 * 256, 256 * 256

    @staticmethod
    def _extract_target_psnr(plan: PlanDocument) -> float:
        """Extract target PSNR from the plan's demands or task description."""
        # Look for PSNR target in the task text
        task = plan.task.lower()
        for phrase in ["psnr", "psnr >=", "psnr≥", "target"]:
            idx = task.find(phrase)
            if idx >= 0:
                # Try to parse a number after
                rest = task[idx + len(phrase):].strip()
                rest = rest.lstrip(">=≥: ")
                num_str = ""
                for ch in rest:
                    if ch.isdigit() or ch == ".":
                        num_str += ch
                    else:
                        break
                if num_str:
                    try:
                        return float(num_str)
                    except ValueError:
                        pass
        # Default target
        return 20.0

    @staticmethod
    def _compute_measurement_snr(
        plan: PlanDocument,
        system_elements: dict[str, Any] | None,
    ) -> float:
        """Compute measurement SNR in dB from source + detector + noise parameters."""
        if system_elements:
            source = system_elements.get("source", {})
            detector = system_elements.get("detector", {})

            qe = detector.get("quantum_efficiency", 0.7)
            read_noise = detector.get("read_noise", 2.0)
            dark_current = detector.get("dark_current", 0.1)
            t_exp = detector.get("exposure_time", 1.0)

            # Estimate photon count from source power
            power_mw = source.get("power_mw", source.get("power", 50.0))
            if isinstance(power_mw, str):
                power_mw = float("".join(c for c in str(power_mw) if c.isdigit() or c == ".") or "50")
            n_photon = max(power_mw * 100, 1000)  # simplified estimate

            signal = qe * n_photon
            noise = math.sqrt(qe * n_photon + read_noise ** 2 + dark_current * t_exp)
            snr_linear = signal / max(noise, 1e-10)
            return 10.0 * math.log10(max(snr_linear, 1.0))

        # Fallback: estimate from noise spec in plan
        if isinstance(plan.action, ForwardAction):
            for el in plan.action.elements:
                for noise in el.noise:
                    if noise.type == "poisson":
                        n_photon = noise.parameters.get("mean_photons",
                                   noise.parameters.get("I0", 10000))
                        return 10.0 * math.log10(max(math.sqrt(n_photon), 1.0))
                    elif noise.type == "gaussian":
                        sigma = noise.parameters.get("sigma",
                                noise.parameters.get("sigma_electrons", 1.0))
                        if sigma > 0:
                            return 10.0 * math.log10(max(1.0 / sigma, 1.0)) + 20.0
        return 30.0  # default estimate

    @staticmethod
    def _estimate_conditioning_factor(plan: PlanDocument) -> float:
        """Estimate C_M = 10*log10((n*||x||_inf^2/||x||^2) * kappa(H)^-2).

        This is a conservative estimate based on modality class.
        """
        modality = plan.modality.lower()
        # Well-conditioned: CT, MRI, OCT
        if any(m in modality for m in ["ct", "mri", "oct", "ultrasound"]):
            return -5.0
        # Moderately conditioned: CASSI, CACTI, SIM
        elif any(m in modality for m in ["cassi", "cacti", "sim", "ptych"]):
            return -10.0
        # Poorly conditioned: lensless, DOT
        elif any(m in modality for m in ["lensless", "dot", "diffuse"]):
            return -20.0
        return -10.0  # default

    @staticmethod
    def _compute_mismatch_degradation(
        plan: PlanDocument,
        system_elements: dict[str, Any] | None,
    ) -> tuple[float, str, str]:
        """Compute total mismatch degradation from calibration tolerances.

        Returns (delta_psnr_total, dominant_param, details_string).
        """
        if system_elements and "calibration" in system_elements:
            cal = system_elements["calibration"]
            params = cal.get("parameters", [])
            if params:
                squared_sum = 0.0
                dominant_param = ""
                max_delta = 0.0
                details_parts = []

                for p in params:
                    name = p.get("name", "unknown")
                    sensitivity = p.get("sensitivity", 1.0)
                    tolerance = p.get("tolerance", 0.1)
                    # Parse sensitivity if it's a string like "13.98dB/px"
                    if isinstance(sensitivity, str):
                        sensitivity = float(
                            "".join(c for c in sensitivity.split("dB")[0]
                                    if c.isdigit() or c == "." or c == "-") or "1"
                        )
                    if isinstance(tolerance, str):
                        tolerance = float(
                            "".join(c for c in tolerance.split("px")[0].split("%")[0]
                                    if c.isdigit() or c == "." or c == "-") or "0.1"
                        )

                    delta = abs(sensitivity * tolerance)
                    squared_sum += delta ** 2
                    details_parts.append(f"{name}: {delta:.2f} dB")
                    if delta > max_delta:
                        max_delta = delta
                        dominant_param = name

                total = math.sqrt(squared_sum)
                return total, dominant_param, "; ".join(details_parts)

        # Fallback: estimate from mismatch specs in plan
        if isinstance(plan.action, ForwardAction):
            high_severity = sum(1 for el in plan.action.elements
                               for m in el.mismatch if m.severity == "high")
            medium_severity = sum(1 for el in plan.action.elements
                                 for m in el.mismatch if m.severity == "medium")
            total = high_severity * 3.0 + medium_severity * 1.5
            dominant = "unknown (no calibration data)"
            return total, dominant, f"{high_severity} high + {medium_severity} medium mismatches"

        return 2.0, "unknown", "default estimate (no calibration data)"

    @staticmethod
    def _extract_recovery_ratio(system_elements: dict[str, Any] | None) -> float:
        """Extract autonomous recovery ratio from calibration spec."""
        if system_elements and "calibration" in system_elements:
            return system_elements["calibration"].get("autonomous_recovery_rate", 0.85)
        return 0.85  # default from flagship paper


# ── Cost Estimator ────────────────────────────────────────────────────────────

class CostEstimator:
    """Estimate hardware cost from system_elements sub-fields."""

    # Rough cost database (USD) by element type
    _COST_DB: dict[str, float] = {
        # Sources
        "x_ray_tube": 50000, "laser": 5000, "broadband_led": 300,
        "broadband_lamp": 500, "led": 200, "rf_coil": 8000,
        "superconducting_magnet": 500000, "transducer_array": 15000,
        "electron_gun": 100000,
        # Optics
        "coded_aperture": 500, "coded_aperture_mask": 500,
        "prism_disperser": 800, "amici_prism": 800,
        "relay_lens": 500, "objective_lens": 2000, "slm": 8000,
        "diffuser": 100, "phase_diffuser": 150,
        "receive_coil_array": 20000, "gradient_system": 100000,
        # Detectors
        "cmos": 2500, "ccd": 3000, "scmos": 15000,
        "photon_counter": 20000, "flat_panel": 30000,
        "scintillator_photodiode": 25000, "quadrature_receiver": 50000,
        "piezo_array": 10000,
        # Calibration
        "spectral_lamp": 200, "flat_field_target": 100,
        "point_source": 50, "calibration_phantom": 500,
    }

    @classmethod
    def estimate(cls, system_elements: dict[str, Any] | None) -> CostEstimate:
        """Produce an itemized cost estimate from system_elements."""
        if not system_elements:
            return CostEstimate(total_usd=0, confidence="unknown")

        items: list[CostItem] = []

        # Source
        source = system_elements.get("source", {})
        if source:
            src_type = source.get("type", "unknown").lower().replace(" ", "_")
            cost = cls._COST_DB.get(src_type, 1000)
            items.append(CostItem(
                category="source", item=src_type,
                cost_usd=cost, confidence="estimated",
            ))

        # Optics
        optics = system_elements.get("optics", [])
        if isinstance(optics, list):
            for opt in optics:
                elem = opt.get("element", opt.get("type", "unknown")).lower().replace(" ", "_")
                cost = cls._COST_DB.get(elem, 500)
                items.append(CostItem(
                    category="optics", item=elem,
                    cost_usd=cost, confidence="estimated",
                ))
        elif isinstance(optics, dict):
            elem = optics.get("type", "unknown").lower().replace(" ", "_")
            cost = cls._COST_DB.get(elem, 500)
            items.append(CostItem(
                category="optics", item=elem,
                cost_usd=cost, confidence="estimated",
            ))

        # Detector
        detector = system_elements.get("detector", {})
        if detector:
            det_type = detector.get("type", "cmos").lower().replace(" ", "_")
            cost = cls._COST_DB.get(det_type, 2500)
            items.append(CostItem(
                category="detector", item=det_type,
                cost_usd=cost, confidence="estimated",
            ))

        # Calibration equipment
        cal = system_elements.get("calibration", {})
        if cal:
            params = cal.get("parameters", [])
            cal_methods = set()
            for p in params:
                method = p.get("correction_method", "")
                if method:
                    cal_methods.add(method)
            for method in cal_methods:
                cost = 200  # default calibration equipment cost
                items.append(CostItem(
                    category="calibration", item=method,
                    cost_usd=cost, confidence="estimated",
                ))

        total = sum(item.cost_usd for item in items)
        return CostEstimate(items=items, total_usd=total, confidence="estimated")


# ── Feasibility Checker ───────────────────────────────────────────────────────

class FeasibilityChecker:
    """Check physical feasibility of each system element."""

    @classmethod
    def check(cls, system_elements: dict[str, Any] | None) -> FeasibilityReport:
        """Assess physical realizability of all system elements."""
        if not system_elements:
            return FeasibilityReport(all_feasible=True)

        items: list[FeasibilityItem] = []

        # Check source
        source = system_elements.get("source", {})
        if source:
            items.append(cls._check_source(source))

        # Check detector
        detector = system_elements.get("detector", {})
        if detector:
            items.append(cls._check_detector(detector))

        # Check optics
        optics = system_elements.get("optics", [])
        if isinstance(optics, list):
            for opt in optics:
                items.append(cls._check_optic(opt))
        elif isinstance(optics, dict):
            items.append(cls._check_optic(optics))

        blocking = [item.element for item in items if not item.feasible]
        return FeasibilityReport(
            items=items,
            all_feasible=len(blocking) == 0,
            blocking_elements=blocking,
        )

    @staticmethod
    def _check_source(source: dict[str, Any]) -> FeasibilityItem:
        src_type = source.get("type", "unknown")
        # Basic realizability checks
        power = source.get("power_mw", source.get("power", 0))
        if isinstance(power, str):
            power = float("".join(c for c in str(power) if c.isdigit() or c == ".") or "0")
        if power > 1e6:  # > 1 MW is impractical for most lab settings
            return FeasibilityItem(
                element=f"source ({src_type})", feasible=False,
                reason=f"Power {power} mW exceeds practical limits",
            )
        return FeasibilityItem(
            element=f"source ({src_type})", feasible=True,
            lead_time="off-the-shelf" if power < 10000 else "custom order",
        )

    @staticmethod
    def _check_detector(detector: dict[str, Any]) -> FeasibilityItem:
        det_type = detector.get("type", "unknown")
        pixels = detector.get("pixels", detector.get("resolution", [256, 256]))
        if isinstance(pixels, list) and len(pixels) >= 2:
            total_pixels = pixels[0] * pixels[1]
            if total_pixels > 100_000_000:  # > 100 MP
                return FeasibilityItem(
                    element=f"detector ({det_type})", feasible=False,
                    reason=f"Resolution {pixels} ({total_pixels/1e6:.0f} MP) exceeds state-of-the-art",
                )
        qe = detector.get("quantum_efficiency", 0.5)
        if qe > 1.0:
            return FeasibilityItem(
                element=f"detector ({det_type})", feasible=False,
                reason=f"Quantum efficiency {qe} > 1.0 is physically impossible",
            )
        return FeasibilityItem(
            element=f"detector ({det_type})", feasible=True,
            lead_time="off-the-shelf",
        )

    @staticmethod
    def _check_optic(optic: dict[str, Any]) -> FeasibilityItem:
        elem = optic.get("element", optic.get("type", "unknown"))
        specs = optic.get("specs", optic)
        # Check NA for objectives
        na = specs.get("NA", specs.get("na", None))
        if na is not None and float(na) > 1.7:
            return FeasibilityItem(
                element=f"optics ({elem})", feasible=False,
                reason=f"NA={na} exceeds maximum achievable (~1.7 for oil immersion)",
            )
        return FeasibilityItem(
            element=f"optics ({elem})", feasible=True,
            lead_time="off-the-shelf",
        )


# ── LLM System Prompt ─────────────────────────────────────────────────────────

_SYSTEM_JUDGE = """\
You are an expert judge evaluating imaging system plans.
You receive a structured plan (markdown text) along with deterministic Triad gate
results (G1: recoverability, G2: carrier budget, G3: operator mismatch) and must
assess overall feasibility.

The Triad gates have already been evaluated deterministically. Your role is to:
1. Review the gate results and provide a qualitative summary
2. Identify any issues NOT captured by the gates (e.g., convergence, stability)
3. Provide specific redesign guidance if the plan fails

Return a JSON object with EXACTLY this structure:

{
  "feasible": true or false,
  "confidence": 0.0-1.0,
  "summary": "<one paragraph judgment incorporating Triad gate results>",
  "issues": [
    {
      "category": "<physics|recoverability|carrier_budget|mismatch|convergence|algorithm|cost|feasibility|other>",
      "severity": "<warning|critical>",
      "element_id": "<element or step id, or empty>",
      "description": "<what is wrong>",
      "suggestion": "<how to fix>"
    }
  ],
  "convergence_likely": true or false or null,
  "mismatch_handled": true or false or null,
  "redesign_prompt": "<if not feasible: concise instruction for redesign, else empty string>"
}

Judgment rules:
- feasible=false if ANY Triad gate fails (G1, G2, or G3)
- feasible=false if algorithm diverges or critical physics issue
- feasible=true with warnings for: suboptimal parameters, minor issues
- Be rigorous but constructive. If feasible=false, redesign_prompt must reference
  the specific failing gate and its design_action.
"""


# ── Judge Agent ───────────────────────────────────────────────────────────────

class JudgeAgent(BaseAgent):
    """Evaluates the feasibility of a PlanDocument via Triad gates + LLM assessment."""

    def __init__(self):
        super().__init__(model=JUDGE_MODEL)

    def judge(self, plan: PlanDocument, system_elements: dict[str, Any] | None = None) -> JudgmentResult:
        """Return a JudgmentResult with Triad gate evaluation, cost, and feasibility.

        Parameters
        ----------
        plan : PlanDocument
            The plan to evaluate.
        system_elements : dict, optional
            Hardware parameters (source, optics, detector, calibration).
            If None, the Triad gates use estimates from the plan's noise/mismatch fields.
        """
        # ── Stage 2: Triad gate evaluation (primary) ──
        triad = TriadGateEvaluator.evaluate_all(plan, system_elements)

        # ── Stage 3: Cost + feasibility ──
        cost = CostEstimator.estimate(system_elements)
        feasibility = FeasibilityChecker.check(system_elements)

        # ── LLM qualitative assessment ──
        period_hint = self._period_context(plan)
        triad_summary = self._format_triad_for_llm(triad)
        cost_summary = f"Estimated cost: ${cost.total_usd:,.0f}" if cost.total_usd > 0 else "Cost: unknown"
        feas_summary = "All elements feasible" if feasibility.all_feasible else (
            f"Blocking elements: {', '.join(feasibility.blocking_elements)}"
        )

        user_content = f"""Period: **{plan.period}**
Modality: **{plan.modality}**
Iteration: {plan.iteration}

{period_hint}

## Triad Gate Results (Deterministic)

{triad_summary}

## System Assessment

{cost_summary}
{feas_summary}

## Plan to Evaluate

```markdown
{plan.to_markdown()}
```
"""
        raw = self.complete_json(
            system=_SYSTEM_JUDGE,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=4096,
        )

        issues = [JudgmentIssue(**i) for i in raw.get("issues", [])]

        # Add critical issues for failed Triad gates
        for gate_result in [triad.gate1, triad.gate2, triad.gate3]:
            if not gate_result.passed:
                issues.append(JudgmentIssue(
                    category=gate_result.gate,
                    severity="critical",
                    description=gate_result.detail,
                    suggestion=gate_result.design_action,
                ))

        # Add feasibility issues
        for item in feasibility.items:
            if not item.feasible:
                issues.append(JudgmentIssue(
                    category="feasibility",
                    severity="critical",
                    element_id=item.element,
                    description=f"Element not feasible: {item.reason}",
                    suggestion=f"Replace with feasible alternative",
                ))

        # Overall feasibility: Triad gates + LLM + feasibility
        overall_feasible = (
            triad.triad_passed
            and feasibility.all_feasible
            and raw.get("feasible", True)
        )

        result = JudgmentResult(
            feasible=overall_feasible,
            confidence=raw.get("confidence", 0.8),
            summary=raw.get("summary", ""),
            issues=issues,
            redesign_prompt=raw.get("redesign_prompt", ""),
            # Triad results
            triad=triad,
            # Cost and feasibility
            cost=cost,
            feasibility_report=feasibility,
            # Legacy fields
            snr_estimate_db=triad.gate2.score_db,
            cost_estimate_usd=cost.total_usd if cost.total_usd > 0 else None,
            budget_ok=cost.total_usd < 1_000_000 if cost.total_usd > 0 else None,
            convergence_likely=raw.get("convergence_likely"),
            mismatch_handled=triad.gate3.passed,
        )

        verdict = "PASS" if result.feasible else "FAIL"
        print(f"[JudgeAgent] Verdict: {verdict} (confidence={result.confidence:.2f})")
        print(f"[JudgeAgent] Triad: G1={'PASS' if triad.gate1.passed else 'FAIL'} "
              f"({triad.gate1.margin_db:+.1f} dB), "
              f"G2={'PASS' if triad.gate2.passed else 'FAIL'} "
              f"({triad.gate2.margin_db:+.1f} dB), "
              f"G3={'PASS' if triad.gate3.passed else 'FAIL'} "
              f"({triad.gate3.margin_db:+.1f} dB)")
        if cost.total_usd > 0:
            print(f"[JudgeAgent] Cost: ${cost.total_usd:,.0f}")
        if not feasibility.all_feasible:
            print(f"[JudgeAgent] Feasibility: BLOCKED by {feasibility.blocking_elements}")
        if result.critical_issues:
            print(f"[JudgeAgent] Critical issues ({len(result.critical_issues)}):")
            for issue in result.critical_issues:
                print(f"  - [{issue.category}] {issue.description[:100]}")

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _period_context(self, plan: PlanDocument) -> str:
        if plan.period == "forward":
            return (
                "Focus on:\n"
                "1. Triad gate results (G1: recoverability, G2: carrier budget, G3: mismatch)\n"
                "2. Whether the Triad gates have sufficient margin\n"
                "3. Any issues NOT captured by the deterministic gates\n"
                "4. Physical correctness of each element"
            )
        else:
            return (
                "Focus on:\n"
                "1. Whether the algorithm converges without pre-training\n"
                "2. Whether all critical mismatch sources are corrected\n"
                "3. Mathematical consistency of algorithm steps\n"
                "4. Numerical stability (condition numbers, step sizes)"
            )

    @staticmethod
    def _format_triad_for_llm(triad: TriadReport) -> str:
        """Format Triad results as text for the LLM prompt."""
        lines = [
            f"Gate 1 (Recoverability): {'PASS' if triad.gate1.passed else 'FAIL'} "
            f"[margin: {triad.gate1.margin_db:+.1f} dB]",
            f"  {triad.gate1.detail}",
            "",
            f"Gate 2 (Carrier Budget): {'PASS' if triad.gate2.passed else 'FAIL'} "
            f"[margin: {triad.gate2.margin_db:+.1f} dB]",
            f"  {triad.gate2.detail}",
            "",
            f"Gate 3 (Operator Mismatch): {'PASS' if triad.gate3.passed else 'FAIL'} "
            f"[margin: {triad.gate3.margin_db:+.1f} dB]",
            f"  {triad.gate3.detail}",
            "",
            f"Dominant gate: {triad.dominant_gate}",
            f"Overall: {'ALL PASS' if triad.triad_passed else 'FAIL'}",
        ]
        if triad.expected_deployment_psnr_db is not None:
            lines.append(f"Expected deployment PSNR: {triad.expected_deployment_psnr_db:.1f} dB")
        return "\n".join(lines)
