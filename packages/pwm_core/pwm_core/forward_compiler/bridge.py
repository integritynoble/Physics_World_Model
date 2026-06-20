"""Bridge: build ForwardModels from modality templates or digital-twin specs.

The agent's NL/equation reasoning produces a modality name + dimensions; this
bridge turns that into the concrete primitive pipeline. Array assets (e.g. the
coded-aperture mask) are passed in directly.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pwm_core.forward_compiler.ir import ForwardModel, Stage


def from_modality(modality: str, *, H: int, W: int, N_bands: int = 1,
                  mask: Optional[np.ndarray] = None,
                  dispersion: Optional[Dict[str, Any]] = None) -> ForwardModel:
    """Return a ForwardModel for a known modality template."""
    modality = modality.lower()
    if modality == "cassi":
        if mask is None:
            mask = np.ones((H, W), dtype=np.float64)
        disp = dispersion or {
            "dispersion_model": "poly",
            "disp_poly_x": [0.0, 1.0],
            "disp_poly_y": [0.0, 0.0],
        }
        return ForwardModel(
            name=f"cassi_{H}x{W}x{N_bands}",
            x_shape=(H, W, N_bands),
            stages=[
                Stage(op="band_shift", params={"dispersion": disp}),
                Stage(op="mask_multiply", params={"mask": np.asarray(mask, dtype=np.float64)}),
                Stage(op="band_sum", params={}),
            ],
            metadata={"modality": "cassi"},
        )
    raise ValueError(f"unknown modality {modality!r}; known templates: ['cassi']")


def from_spec_fields(fields: Dict[str, Any], *,
                     mask: Optional[np.ndarray] = None) -> ForwardModel:
    """Build a ForwardModel from digital-twin spec fields (six_tuple/protocol).

    Recognises the same field layout produced by the optics pwm_bridge.
    """
    modality = (fields.get("spec_type") or fields.get("modality") or "").lower()
    omega = (fields.get("six_tuple") or {}).get("omega", {})
    H = int(omega.get("H", 0)) or int(fields.get("H", 0))
    W = int(omega.get("W", 0)) or int(fields.get("W", 0))
    N_bands = int(omega.get("N_bands", 1) or 1)
    if modality == "cassi":
        pf = fields.get("protocol_fields", {})
        a1 = float(pf.get("disp_a1_nominal", 1.0))
        disp = {"dispersion_model": "poly",
                "disp_poly_x": [0.0, a1], "disp_poly_y": [0.0, 0.0]}
        return from_modality("cassi", H=H, W=W, N_bands=N_bands,
                             mask=mask, dispersion=disp)
    raise ValueError(f"from_spec_fields: unsupported modality {modality!r}")
