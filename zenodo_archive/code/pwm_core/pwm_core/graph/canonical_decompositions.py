"""pwm_core.graph.canonical_decompositions
==========================================

Canonical DAG decomposition registry for the 31 modalities enumerated
in the Finite Primitive Basis Theorem (Table 1).

Each entry records the canonical primitive chain, node/depth counts,
carrier type, fidelity bound, and validation level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pwm_core.graph.ir_types import CanonicalPrimitive

# Shorthand
P = CanonicalPrimitive.P
M = CanonicalPrimitive.M
Pi = CanonicalPrimitive.Pi
F = CanonicalPrimitive.F
C = CanonicalPrimitive.C
Sigma = CanonicalPrimitive.Sigma
D = CanonicalPrimitive.D
S = CanonicalPrimitive.S
W = CanonicalPrimitive.W
R = CanonicalPrimitive.R
Lambda = CanonicalPrimitive.Lambda


@dataclass(frozen=True)
class CanonicalDecomposition:
    """Canonical DAG decomposition for one modality.

    Attributes
    ----------
    modality : str
        Modality key (e.g. ``"cassi"``, ``"mri"``).
    dag : str
        Human-readable canonical chain (e.g. ``"M -> W -> Sigma -> D"``).
    primitives : list[CanonicalPrimitive]
        Ordered list of canonical primitives in the chain.
    nodes : int
        Number of canonical nodes in the chain.
    depth : int
        DAG depth (longest path in the canonical graph).
    carrier : str
        Physical carrier type (``"photon"``, ``"electron"``, ``"spin"``,
        ``"acoustic"``, ``"particle"``).
    e_tier2 : str
        Fidelity bound (e.g. ``"<1e-4"``).
    validation_level : str
        One of ``"full"``, ``"held_out"``, ``"exotic"``, ``"template"``.
    """

    modality: str
    dag: str
    primitives: List[CanonicalPrimitive]
    nodes: int
    depth: int
    carrier: str
    e_tier2: str
    validation_level: str


# -----------------------------------------------------------------------
# 31-modality decomposition table (FPT paper Table 1)
# -----------------------------------------------------------------------

CANONICAL_DECOMPOSITIONS: Dict[str, CanonicalDecomposition] = {
    # --- Full-validation modalities (rows 1-7) ---
    "cassi": CanonicalDecomposition(
        "cassi", "M -> W -> Sigma -> D", [M, W, Sigma, D],
        4, 4, "photon", "<1e-4", "full",
    ),
    "cacti": CanonicalDecomposition(
        "cacti", "M -> Sigma -> D", [M, Sigma, D],
        3, 3, "photon", "<1e-4", "full",
    ),
    "spc": CanonicalDecomposition(
        "spc", "M -> Sigma -> D", [M, Sigma, D],
        3, 3, "photon", "<1e-4", "full",
    ),
    "lensless": CanonicalDecomposition(
        "lensless", "C -> D", [C, D],
        2, 2, "photon", "<1e-5", "full",
    ),
    "ptychography": CanonicalDecomposition(
        "ptychography", "M -> P -> D", [M, P, D],
        3, 3, "photon", "4.2e-4", "full",
    ),
    "mri": CanonicalDecomposition(
        "mri", "M -> F -> S -> D", [M, F, S, D],
        4, 4, "spin", "<1e-6", "full",
    ),
    "ct": CanonicalDecomposition(
        "ct", "Pi -> D", [Pi, D],
        2, 2, "xray", "<1e-5", "full",
    ),
    # --- Held-out modalities (rows 8-12) ---
    "oct": CanonicalDecomposition(
        "oct", "P + P -> Sigma -> D", [P, P, Sigma, D],
        4, 3, "photon", "3.8e-4", "held_out",
    ),
    "photoacoustic": CanonicalDecomposition(
        "photoacoustic", "M -> P -> D", [M, P, D],
        3, 3, "acoustic", "7.1e-4", "held_out",
    ),
    "sim": CanonicalDecomposition(
        "sim", "M -> C -> D", [M, C, D],
        3, 3, "photon", "2.5e-4", "held_out",
    ),
    "phase_contrast": CanonicalDecomposition(
        "phase_contrast", "Pi -> P -> M -> D", [Pi, P, M, D],
        4, 4, "xray", "1.2e-3", "held_out",
    ),
    "electron_ptycho": CanonicalDecomposition(
        "electron_ptycho", "M -> P -> D", [M, P, D],
        3, 3, "electron", "5.6e-4", "held_out",
    ),
    # --- Exotic modalities (rows 13-19) ---
    "ghost_imaging": CanonicalDecomposition(
        "ghost_imaging", "M -> Sigma -> D", [M, Sigma, D],
        3, 3, "photon", "1.9e-4", "exotic",
    ),
    "thz_tds": CanonicalDecomposition(
        "thz_tds", "C -> D", [C, D],
        2, 2, "photon", "8.3e-4", "exotic",
    ),
    "compton": CanonicalDecomposition(
        "compton", "M -> R -> D", [M, R, D],
        3, 3, "xray", "6.7e-3", "exotic",
    ),
    "raman": CanonicalDecomposition(
        "raman", "M -> R -> D", [M, R, D],
        3, 3, "photon", "4.1e-3", "exotic",
    ),
    "fluorescence": CanonicalDecomposition(
        "fluorescence", "M -> R -> D", [M, R, D],
        3, 3, "photon", "3.8e-3", "exotic",
    ),
    "dot": CanonicalDecomposition(
        "dot", "M -> R -> P -> R -> D", [M, R, P, R, D],
        5, 5, "photon", "8.9e-3", "exotic",
    ),
    "brillouin": CanonicalDecomposition(
        "brillouin", "M -> R -> D", [M, R, D],
        3, 3, "photon", "5.2e-3", "exotic",
    ),
    # --- Template modalities (rows 20-31) ---
    "spect": CanonicalDecomposition(
        "spect", "Pi -> S -> D", [Pi, S, D],
        3, 3, "xray", "<1e-4", "template",
    ),
    "pet": CanonicalDecomposition(
        "pet", "Pi -> D", [Pi, D],
        2, 2, "xray", "<1e-4", "template",
    ),
    "ultrasound": CanonicalDecomposition(
        "ultrasound", "P -> D", [P, D],
        2, 2, "acoustic", "<1e-4", "template",
    ),
    "sted": CanonicalDecomposition(
        "sted", "M -> C -> D", [M, C, D],
        3, 3, "photon", "<1e-3", "template",
    ),
    "palm_storm": CanonicalDecomposition(
        "palm_storm", "M -> C -> D", [M, C, D],
        3, 3, "photon", "<1e-3", "template",
    ),
    "tirf": CanonicalDecomposition(
        "tirf", "P -> C -> D", [P, C, D],
        3, 3, "photon", "<1e-3", "template",
    ),
    "doppler_us": CanonicalDecomposition(
        "doppler_us", "P -> D", [P, D],
        2, 2, "acoustic", "<1e-3", "template",
    ),
    "elastography": CanonicalDecomposition(
        "elastography", "P -> P -> D", [P, P, D],
        3, 3, "acoustic", "<1e-3", "template",
    ),
    "dexa": CanonicalDecomposition(
        "dexa", "M -> Pi -> D", [M, Pi, D],
        3, 3, "xray", "<1e-4", "template",
    ),
    "sem": CanonicalDecomposition(
        "sem", "M -> D", [M, D],
        2, 2, "electron", "<1e-4", "template",
    ),
    "tem": CanonicalDecomposition(
        "tem", "M -> C -> D", [M, C, D],
        3, 3, "electron", "<1e-4", "template",
    ),
    "muon_tomography": CanonicalDecomposition(
        "muon_tomography", "R -> Pi -> D", [R, Pi, D],
        3, 3, "particle", "<1e-2", "template",
    ),
    # --- Modalities requiring Lambda (Transform) primitive ---
    "ct_polychromatic": CanonicalDecomposition(
        "ct_polychromatic", "Pi -> Lambda -> D", [Pi, Lambda, D],
        3, 3, "xray", "<1e-3", "template",
    ),
    "mri_phase_wrapped": CanonicalDecomposition(
        "mri_phase_wrapped", "M -> F -> S -> Lambda -> D", [M, F, S, Lambda, D],
        5, 5, "spin", "<1e-3", "template",
    ),
    "cbct": CanonicalDecomposition(
        "cbct", "Pi -> Lambda -> D", [Pi, Lambda, D],
        3, 3, "xray", "<1e-3", "template",
    ),
    "proton_therapy": CanonicalDecomposition(
        "proton_therapy", "Lambda -> Pi -> D", [Lambda, Pi, D],
        3, 3, "particle", "<1e-2", "template",
    ),
    "fluorescence_saturated": CanonicalDecomposition(
        "fluorescence_saturated", "M -> R -> Lambda -> D", [M, R, Lambda, D],
        4, 4, "photon", "<1e-3", "template",
    ),
}


# -----------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------

# Maximum allowed node count and depth (FPB bounds)
N_MAX = 20
D_MAX = 10


def validate_decomposition(
    modality: str,
    canonical_chain: List[Optional[CanonicalPrimitive]],
) -> Tuple[bool, List[str]]:
    """Validate a compiled graph's canonical chain against the registry.

    Parameters
    ----------
    modality : str
        Modality key to look up in ``CANONICAL_DECOMPOSITIONS``.
    canonical_chain : list
        The list returned by ``GraphOperator.to_canonical()``.

    Returns
    -------
    (passed, failures) : tuple[bool, list[str]]
        Whether validation passed and a list of failure messages.
    """
    failures: List[str] = []

    if modality not in CANONICAL_DECOMPOSITIONS:
        failures.append(f"Unknown modality '{modality}' (not in registry)")
        return (False, failures)

    expected = CANONICAL_DECOMPOSITIONS[modality]

    # Filter out None (sources, noise, corrections)
    actual = [c for c in canonical_chain if c is not None]

    if actual != expected.primitives:
        failures.append(
            f"Canonical chain mismatch: expected {[p.name for p in expected.primitives]}, "
            f"got {[p.name for p in actual]}"
        )

    if len(actual) > N_MAX:
        failures.append(
            f"Node count {len(actual)} exceeds N_max={N_MAX}"
        )

    # All nodes must map to known canonical primitives (already guaranteed by enum)
    for i, cid in enumerate(actual):
        if not isinstance(cid, CanonicalPrimitive):
            failures.append(f"Node {i}: {cid!r} is not a CanonicalPrimitive")

    return (len(failures) == 0, failures)
