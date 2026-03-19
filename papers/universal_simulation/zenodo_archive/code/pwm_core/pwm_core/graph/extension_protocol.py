"""pwm_core.graph.extension_protocol
=====================================

5-step formal extension protocol from Section 7 of the FPT paper.

Implements:
- ExtensionProposal dataclass for new canonical primitive proposals
- validate_extension() to check all 5 criteria
- basis_growth_curve() to compute K-vs-N saturation curve
- is_saturated() to check for basis saturation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from pwm_core.graph.canonical_decompositions import CANONICAL_DECOMPOSITIONS
from pwm_core.graph.ir_types import CanonicalPrimitive


@dataclass
class ExtensionProposal:
    """A proposal to add a new canonical primitive to B_lib.

    Attributes
    ----------
    primitive_id : str
        Implementation-level primitive_id of the candidate.
    canonical_name : str
        Proposed canonical name (e.g. ``"Q"`` for some new operator).
    forward_adjoint_validated : bool
        Step 1: forward()/adjoint() pass the randomized dot-product test.
    representation_gap : float
        Step 2: min_G e_tier2 > epsilon when the new primitive is absent.
    error_with_primitive : float
        Step 3: e_tier2 < epsilon when the new primitive is included.
    modalities_requiring : list[str]
        Step 4: list of modalities that need this primitive (need >= 2).
    closure_test_passed : bool
        Step 5: backward-compatible; all existing decompositions still valid.
    """

    primitive_id: str
    canonical_name: str
    forward_adjoint_validated: bool = False
    representation_gap: float = 0.0
    error_with_primitive: float = float("inf")
    modalities_requiring: List[str] = field(default_factory=list)
    closure_test_passed: bool = False


def validate_extension(
    proposal: ExtensionProposal,
    epsilon: float = 0.01,
) -> Tuple[bool, List[str]]:
    """Validate a canonical primitive extension proposal.

    All 5 criteria must pass for the extension to be accepted.

    Parameters
    ----------
    proposal : ExtensionProposal
        The extension proposal to validate.
    epsilon : float
        Fidelity threshold (default 0.01 = 1%).

    Returns
    -------
    (passed, failures) : tuple[bool, list[str]]
        Whether all criteria passed and a list of failure messages.
    """
    failures: List[str] = []

    if not proposal.forward_adjoint_validated:
        failures.append("Step 1: forward/adjoint not validated")

    if proposal.representation_gap <= epsilon:
        failures.append(
            f"Step 2: representation gap {proposal.representation_gap:.4f} "
            f"<= eps={epsilon} (existing basis suffices)"
        )

    if proposal.error_with_primitive > epsilon:
        failures.append(
            f"Step 3: error with primitive {proposal.error_with_primitive:.4f} "
            f"> eps={epsilon} (primitive does not close the gap)"
        )

    if len(proposal.modalities_requiring) < 2:
        failures.append(
            f"Step 4: only {len(proposal.modalities_requiring)} "
            f"modalities need it (need >= 2)"
        )

    if not proposal.closure_test_passed:
        failures.append(
            "Step 5: closure test failed (backward compatibility broken)"
        )

    return (len(failures) == 0, failures)


def basis_growth_curve(
    registration_order: Optional[List[str]] = None,
) -> List[Tuple[int, int]]:
    """Compute the K-vs-N basis growth curve.

    Given a modality registration order, returns [(N, K)] where N is the
    number of modalities registered so far, K is the number of distinct
    canonical primitives needed.

    Parameters
    ----------
    registration_order : list[str], optional
        Ordered list of modality keys. If None, uses the natural order
        from CANONICAL_DECOMPOSITIONS.

    Returns
    -------
    list[tuple[int, int]]
        List of (N, K) pairs.
    """
    if registration_order is None:
        registration_order = list(CANONICAL_DECOMPOSITIONS.keys())

    seen: Set[CanonicalPrimitive] = set()
    curve: List[Tuple[int, int]] = []

    for i, modality in enumerate(registration_order, 1):
        if modality in CANONICAL_DECOMPOSITIONS:
            decomp = CANONICAL_DECOMPOSITIONS[modality]
            seen.update(decomp.primitives)
        curve.append((i, len(seen)))

    return curve


def is_saturated(
    curve: Optional[List[Tuple[int, int]]] = None,
    window: int = 5,
) -> bool:
    """Check if the basis has saturated.

    Returns True if K (number of distinct canonical primitives) has not
    increased in the last ``window`` modalities.

    Parameters
    ----------
    curve : list[tuple[int, int]], optional
        Output of ``basis_growth_curve()``. Computed from defaults if None.
    window : int
        Number of consecutive modalities with no growth to declare saturation.

    Returns
    -------
    bool
        True if the basis is saturated.
    """
    if curve is None:
        curve = basis_growth_curve()

    if len(curve) < window + 1:
        return False

    # Check if K is constant over the last `window` entries
    final_k = curve[-1][1]
    for _, k in curve[-window:]:
        if k != final_k:
            return False
    return True
