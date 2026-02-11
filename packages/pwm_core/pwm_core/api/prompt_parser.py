"""pwm_core.api.prompt_parser
==============================

Prompt -> ExperimentSpec parser.

Extracts modality, photon budget, execution mode, and solver preferences
from a natural-language prompt string, then assembles a validated
ExperimentSpec (v0.2.1).
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pwm_core.core.enums import ExecutionMode
from pwm_core.api.types import (
    ExperimentSpec,
    ExperimentInput,
    ExperimentStates,
    InputMode,
    PhysicsState,
    BudgetState,
    TaskState,
    TaskKind,
    ReconSpec,
    ReconPortfolio,
    SolverSpec,
    OperatorInput,
    OperatorKind,
    OperatorParametric,
)


# ---------------------------------------------------------------------------
# Modality keyword map (26 modalities)
# ---------------------------------------------------------------------------

_MODALITY_KEYWORDS: Dict[str, List[str]] = {
    "widefield": ["widefield", "wide-field", "epi-fluorescence"],
    "widefield_lowdose": ["low-dose", "low dose", "lowdose", "photon-starved"],
    "confocal_livecell": ["confocal live", "live-cell confocal", "live cell confocal"],
    "confocal_3d": ["confocal 3d", "3d confocal", "confocal z-stack"],
    "sim": ["structured illumination", "sim "],
    "cassi": ["cassi", "coded aperture spectral"],
    "spc": ["single-pixel", "single pixel", "spc"],
    "cacti": ["cacti", "snapshot compressive"],
    "lensless": ["lensless", "lens-less", "mask-based"],
    "lightsheet": ["light-sheet", "light sheet", "lightsheet", "spim"],
    "ct": [" ct ", "computed tomography", "radon", "sinogram"],
    "mri": [" mri ", "magnetic resonance", "k-space", "kspace"],
    "ptychography": ["ptychograph"],
    "holography": ["hologra", "off-axis"],
    "nerf": ["nerf", "neural radiance"],
    "gaussian_splatting": ["gaussian splat", "3dgs", "3d gaussian"],
    "matrix": ["matrix", "generic linear"],
    "panorama": ["panoram", "multifocal"],
    "light_field": ["light field", "lightfield", "plenoptic"],
    "integral": ["integral imaging", "lenslet"],
    "phase_retrieval": ["phase retrieval", "cdi"],
    "flim": ["flim", "fluorescence lifetime"],
    "photoacoustic": ["photoacoustic", "optoacoustic"],
    "oct": [" oct ", "optical coherence tomography"],
    "fpm": ["fourier ptychograph", " fpm "],
    "dot": ["diffuse optical", " dot "],
}

# ---------------------------------------------------------------------------
# Photon budget map
# ---------------------------------------------------------------------------

_BUDGET_KEYWORDS: Dict[str, float] = {
    "ultra-low": 100.0,
    "very low": 500.0,
    "low dose": 1000.0,
    "low-dose": 1000.0,
    "moderate": 10000.0,
    "bright": 100000.0,
    "high snr": 100000.0,
    "photon-rich": 100000.0,
}

# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

_MODE_KEYWORDS: Dict[ExecutionMode, List[str]] = {
    ExecutionMode.simulate: ["simulat", "forward model", "generate measurement"],
    ExecutionMode.invert: ["reconstruct", "inver", "recover", "solve", "deconvolv"],
    ExecutionMode.calibrate: ["calibrat", "fit operator", "estimate parameter"],
}

# ---------------------------------------------------------------------------
# Solver detection
# ---------------------------------------------------------------------------

_SOLVER_KEYWORDS: Dict[str, List[str]] = {
    "fista": ["fista"],
    "admm": ["admm"],
    "lsq": ["least square", "lsq"],
    "gap_tv": ["gap-tv", "gap_tv"],
    "pgd": ["proximal gradient", "pgd"],
    "rl": ["richardson-lucy", "rl deconv"],
    "red": [" red ", "regularization by denoising"],
}


# ---------------------------------------------------------------------------
# Internal parsed fields (renamed from ParsedPrompt)
# ---------------------------------------------------------------------------


class _ParsedFields:
    """Internal result of keyword extraction from a natural-language prompt.

    Attributes
    ----------
    modality : str or None
        Detected modality key.
    mode : ExecutionMode
        Detected execution mode.
    photon_budget : float or None
        Detected photon budget.
    solver_ids : list[str]
        Detected solver preferences.
    raw : str
        Original prompt.
    """

    def __init__(
        self,
        modality: Optional[str],
        mode: ExecutionMode,
        photon_budget: Optional[float],
        solver_ids: List[str],
        raw: str,
    ) -> None:
        self.modality = modality
        self.mode = mode
        self.photon_budget = photon_budget
        self.solver_ids = solver_ids
        self.raw = raw

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "mode": self.mode.value,
            "photon_budget": self.photon_budget,
            "solver_ids": self.solver_ids,
        }


# Backward-compatible alias
ParsedPrompt = _ParsedFields


# ---------------------------------------------------------------------------
# ExecutionMode -> TaskKind / InputMode mapping
# ---------------------------------------------------------------------------

_MODE_TO_TASK: Dict[ExecutionMode, TaskKind] = {
    ExecutionMode.simulate: TaskKind.simulate_recon_analyze,
    ExecutionMode.invert: TaskKind.reconstruct_only,
    ExecutionMode.calibrate: TaskKind.calibrate_and_reconstruct,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_prompt_raw(prompt: str) -> _ParsedFields:
    """Parse a prompt into raw keyword-extracted fields (backward compat).

    Parameters
    ----------
    prompt : str
        User prompt describing the imaging task.

    Returns
    -------
    _ParsedFields
        Parsed result with modality, mode, budget, and solver preferences.
    """
    text = prompt.lower()

    # 1. Modality detection
    modality = None
    best_pos = len(text) + 1
    for mod_key, keywords in _MODALITY_KEYWORDS.items():
        for kw in keywords:
            pos = text.find(kw)
            if pos != -1 and pos < best_pos:
                modality = mod_key
                best_pos = pos

    # 2. Photon budget
    photon_budget = None
    for budget_kw, budget_val in _BUDGET_KEYWORDS.items():
        if budget_kw in text:
            photon_budget = budget_val
            break

    # Try to extract explicit number: "N photons"
    photon_match = re.search(r"(\d+(?:\.\d+)?(?:e\d+)?)\s*photons?", text)
    if photon_match:
        try:
            photon_budget = float(photon_match.group(1))
        except ValueError:
            pass

    # 3. Mode detection
    mode = ExecutionMode.invert  # default
    for exec_mode, keywords in _MODE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                mode = exec_mode
                break

    # 4. Solver preference
    solver_ids: List[str] = []
    for solver_id, keywords in _SOLVER_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                solver_ids.append(solver_id)
                break

    return _ParsedFields(
        modality=modality,
        mode=mode,
        photon_budget=photon_budget,
        solver_ids=solver_ids,
        raw=prompt,
    )


def parse_prompt(prompt: str) -> ExperimentSpec:
    """Parse a natural-language prompt into a validated ExperimentSpec.

    Parameters
    ----------
    prompt : str
        User prompt describing the imaging task.

    Returns
    -------
    ExperimentSpec
        Fully validated experiment specification (v0.2.1).
    """
    fields = parse_prompt_raw(prompt)

    # Map mode -> TaskKind
    task_kind = _MODE_TO_TASK.get(fields.mode, TaskKind.reconstruct_only)

    # Map mode -> InputMode
    input_mode = (
        InputMode.simulate
        if fields.mode == ExecutionMode.simulate
        else InputMode.measured
    )

    # Build graph_template_id
    template_id = f"{fields.modality}_graph_v2" if fields.modality else None

    # Build operator
    operator = None
    if template_id:
        operator = OperatorInput(
            kind=OperatorKind.parametric,
            parametric=OperatorParametric(operator_id=template_id),
        )

    # Build budget
    budget = None
    if fields.photon_budget is not None:
        budget = BudgetState(
            photon_budget={"max_photons": fields.photon_budget},
        )

    # Build recon portfolio
    portfolio = (
        ReconPortfolio(solvers=[SolverSpec(id=s) for s in fields.solver_ids])
        if fields.solver_ids
        else ReconPortfolio()
    )

    # Build modality string (fallback to "unknown")
    modality_str = fields.modality or "unknown"

    spec = ExperimentSpec(
        id=f"prompt_{uuid.uuid4().hex[:8]}",
        input=ExperimentInput(
            mode=input_mode,
            operator=operator,
        ),
        states=ExperimentStates(
            physics=PhysicsState(
                modality=modality_str,
                graph_template_id=template_id,
            ),
            budget=budget,
            task=TaskState(kind=task_kind),
        ),
        recon=ReconSpec(portfolio=portfolio),
    )

    return spec
