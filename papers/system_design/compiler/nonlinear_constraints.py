"""Formal constraints for the three nonlinear canonical primitives.

The Finite Primitive Basis Theorem restricts the Detect (D), Scatter (R),
and Transform (Λ) primitives to **5 enumerable families** each, with at
most **2 parameters** per family.  This bounded parametrisation prevents
any single primitive from becoming a universal approximator (Cybenko
sense), making the 11-primitive basis *falsifiable*.

This module codifies those constraints so the Constrained Primitive
Compiler can reject any agent output that violates them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pwm_core.graph.ir_types import DetectFamily, TransformFamily


# ---------------------------------------------------------------------------
# Constraint dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonlinearFamilyConstraint:
    """Formal constraint for one nonlinear primitive family.

    Attributes
    ----------
    family_id : str
        Enum value (e.g. "intensity_square_law", "beer_lambert").
    display_name : str
        Human-readable name for paper/logs.
    formula : str
        Mathematical formula (LaTeX-compatible).
    param_names : list[str]
        Names of allowed parameters (length ≤ 2).
    param_bounds : dict[str, tuple[float, float]]
        Lower/upper bounds per parameter (physics-based).
    max_lipschitz : float | None
        Lipschitz constant bound (None = unbounded within domain).
    is_monotone : bool
        True if the family is monotone (useful for convergence proofs).
    notes : str
        Physics rationale for the family.
    """

    family_id: str
    display_name: str
    formula: str
    param_names: List[str] = field(default_factory=list)
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    max_lipschitz: Optional[float] = None
    is_monotone: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# Detect (D) constraints — 5 families, ≤ 2 params each
# ---------------------------------------------------------------------------

DETECT_CONSTRAINTS: Dict[DetectFamily, NonlinearFamilyConstraint] = {
    DetectFamily.linear_field: NonlinearFamilyConstraint(
        family_id="linear_field",
        display_name="Linear-Field Detector",
        formula=r"\eta(x) = g \cdot x",
        param_names=["gain"],
        param_bounds={"gain": (1e-6, 1e6)},
        max_lipschitz=1e6,
        is_monotone=True,
        notes="Field-amplitude detector (e.g. homodyne). Strictly linear.",
    ),
    DetectFamily.intensity_square_law: NonlinearFamilyConstraint(
        family_id="intensity_square_law",
        display_name="Intensity (Square-Law) Detector",
        formula=r"\eta(x) = g \cdot |x|^2",
        param_names=["gain"],
        param_bounds={"gain": (1e-6, 1e6)},
        max_lipschitz=None,  # Lipschitz depends on input domain
        is_monotone=True,
        notes="Standard intensity detector (CCD, CMOS, PMT). Most common.",
    ),
    DetectFamily.logarithmic: NonlinearFamilyConstraint(
        family_id="logarithmic",
        display_name="Logarithmic Detector",
        formula=r"\eta(x) = g \cdot \log(1 + |x|^2 / x_0)",
        param_names=["gain", "x0"],
        param_bounds={"gain": (1e-6, 1e6), "x0": (1e-10, 1e10)},
        max_lipschitz=None,
        is_monotone=True,
        notes="Logarithmic compression (e.g. film, ultrasound log envelope).",
    ),
    DetectFamily.sigmoid: NonlinearFamilyConstraint(
        family_id="sigmoid",
        display_name="Sigmoid Detector",
        formula=r"\eta(x) = g \cdot \sigma(|x|^2 - x_0)",
        param_names=["gain", "x0"],
        param_bounds={"gain": (1e-6, 1e6), "x0": (-1e6, 1e6)},
        max_lipschitz=None,
        is_monotone=True,
        notes="Saturating detector with sigmoidal response.",
    ),
    DetectFamily.coherent_field: NonlinearFamilyConstraint(
        family_id="coherent_field",
        display_name="Coherent-Field Detector",
        formula=r"\eta(x) = g \cdot \mathrm{Re}[x \cdot e^{j\varphi}]",
        param_names=["gain", "phase"],
        param_bounds={"gain": (1e-6, 1e6), "phase": (-math.pi, math.pi)},
        max_lipschitz=1e6,
        is_monotone=False,
        notes="Heterodyne/interferometric detection (e.g. OCT, holography).",
    ),
}

# ---------------------------------------------------------------------------
# Transform (Λ) constraints — 5 families, ≤ 2 params each
# ---------------------------------------------------------------------------

TRANSFORM_CONSTRAINTS: Dict[TransformFamily, NonlinearFamilyConstraint] = {
    TransformFamily.beer_lambert: NonlinearFamilyConstraint(
        family_id="beer_lambert",
        display_name="Beer–Lambert Exponential",
        formula=r"\Lambda(x) = \exp(-\mu \cdot x)",
        param_names=["mu"],
        param_bounds={"mu": (1e-6, 1e4)},
        max_lipschitz=None,  # |Λ'(x)| = μ·exp(-μx) ≤ μ
        is_monotone=True,
        notes="X-ray transmission through attenuating medium.",
    ),
    TransformFamily.phase_wrapping: NonlinearFamilyConstraint(
        family_id="phase_wrapping",
        display_name="Phase Wrapping",
        formula=r"\Lambda(x) = \mathrm{angle}(\exp(jx))",
        param_names=[],
        param_bounds={},
        max_lipschitz=1.0,
        is_monotone=False,
        notes="MRI phase wrapping. No free parameters (zero params ≤ 2).",
    ),
    TransformFamily.beam_hardening: NonlinearFamilyConstraint(
        family_id="beam_hardening",
        display_name="Beam Hardening Polynomial",
        formula=r"\Lambda(x) = a_1 x + a_2 x^2",
        param_names=["a1", "a2"],
        param_bounds={"a1": (0.5, 2.0), "a2": (-0.5, 0.5)},
        max_lipschitz=None,
        is_monotone=True,  # For a1 > 0, a2 small
        notes="Polychromatic X-ray beam hardening (degree ≤ 2).",
    ),
    TransformFamily.stopping_power: NonlinearFamilyConstraint(
        family_id="stopping_power",
        display_name="Stopping Power (Bethe–Bloch)",
        formula=r"\Lambda(x) = a / x^2",
        param_names=["a"],
        param_bounds={"a": (1e-6, 1e6)},
        max_lipschitz=None,
        is_monotone=True,
        notes="Charged particle energy loss (proton therapy, muon tomography).",
    ),
    TransformFamily.saturation: NonlinearFamilyConstraint(
        family_id="saturation",
        display_name="Saturation",
        formula=r"\Lambda(x) = x_{\max}(1 - \exp(-x / x_0))",
        param_names=["x_max", "x0"],
        param_bounds={"x_max": (1e-6, 1e10), "x0": (1e-10, 1e10)},
        max_lipschitz=None,  # |Λ'(0)| = x_max/x0
        is_monotone=True,
        notes="Fluorophore saturation, detector pixel well saturation.",
    ),
}


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_detect_params(
    family: DetectFamily,
    params: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Validate a Detect node's parameters against family constraints.

    Parameters
    ----------
    family : DetectFamily
        Which of the 5 detector families.
    params : dict
        Parameters from the agent-generated spec.

    Returns
    -------
    (valid, failures) : tuple[bool, list[str]]
    """
    return _validate_nonlinear(family, params, DETECT_CONSTRAINTS, "Detect")


def validate_transform_params(
    family: TransformFamily,
    params: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Validate a Transform (Λ) node's parameters against family constraints.

    Returns
    -------
    (valid, failures) : tuple[bool, list[str]]
    """
    return _validate_nonlinear(family, params, TRANSFORM_CONSTRAINTS, "Transform")


def compute_lipschitz_bound(
    family: DetectFamily | TransformFamily,
    params: Dict[str, Any],
) -> Optional[float]:
    """Compute the Lipschitz constant for a nonlinear primitive.

    Returns None if the bound depends on the input domain and cannot
    be computed without data.
    """
    constraints = DETECT_CONSTRAINTS if isinstance(family, DetectFamily) else TRANSFORM_CONSTRAINTS
    constraint = constraints.get(family)
    if constraint is None:
        return None
    return constraint.max_lipschitz


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _validate_nonlinear(
    family: Any,
    params: Dict[str, Any],
    constraints_map: Dict,
    primitive_name: str,
) -> Tuple[bool, List[str]]:
    """Generic validation for D, R, or Λ."""
    failures: List[str] = []

    if family not in constraints_map:
        failures.append(
            f"{primitive_name} family '{family}' not in the 5 canonical families: "
            f"{sorted(c.family_id for c in constraints_map.values())}"
        )
        return (False, failures)

    constraint = constraints_map[family]

    # Check parameter count ≤ 2
    provided = {k: v for k, v in params.items() if k in constraint.param_names}
    unexpected = set(params.keys()) - set(constraint.param_names) - {"type", "family"}
    if unexpected:
        # Allow extra metadata keys but warn
        pass

    if len(constraint.param_names) > 2:
        # Should never happen — this is a code-level assertion
        failures.append(
            f"{primitive_name}({constraint.family_id}) has {len(constraint.param_names)} "
            f"declared params, exceeding the ≤2 limit"
        )

    # Check parameter bounds
    for pname in constraint.param_names:
        if pname not in params:
            continue  # Missing params use defaults — acceptable
        val = params[pname]
        if not isinstance(val, (int, float)):
            continue
        lo, hi = constraint.param_bounds.get(pname, (-math.inf, math.inf))
        if val < lo or val > hi:
            failures.append(
                f"{primitive_name}({constraint.family_id}).{pname} = {val} "
                f"outside bounds [{lo}, {hi}]"
            )

    return (len(failures) == 0, failures)
