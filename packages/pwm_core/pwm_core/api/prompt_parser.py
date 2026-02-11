"""pwm_core.api.prompt_parser
==============================

Prompt -> ExperimentSpec parser.

Extracts modality, photon budget, execution mode, and solver preferences
from a natural-language prompt string.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pwm_core.core.enums import ExecutionMode


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
# Public API
# ---------------------------------------------------------------------------


class ParsedPrompt:
    """Result of parsing a natural-language prompt.

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


def parse_prompt(prompt: str) -> ParsedPrompt:
    """Parse a natural-language prompt into structured fields.

    Parameters
    ----------
    prompt : str
        User prompt describing the imaging task.

    Returns
    -------
    ParsedPrompt
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

    return ParsedPrompt(
        modality=modality,
        mode=mode,
        photon_budget=photon_budget,
        solver_ids=solver_ids,
        raw=prompt,
    )
