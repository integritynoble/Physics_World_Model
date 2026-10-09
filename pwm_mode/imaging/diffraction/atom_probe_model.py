"""Public solver-side atom-probe mass-spectrum forward model.

Scope is deliberately narrow: natural atomic isotope envelopes, singly and
doubly charged ions, a Gaussian mass resolving response, detector efficiency,
and a uniform background. Molecular ions and field-dependent evaporation are
outside this benchmark model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ISOTOPE_PATH = Path(__file__).with_name("nist_isotopes.json")
ISOTOPES = json.loads(ISOTOPE_PATH.read_text())["elements"]
MODE_SIGMA_DA = {"high": 0.025, "nominal": 0.050, "low": 0.090}


def forward_spectrum(case: dict[str, Any], sigma_da: float | None = None) -> np.ndarray:
    """Expected count spectrum on the public mass-to-charge grid."""
    axis = np.asarray(case["mass_to_charge_grid"], dtype=np.float64)
    step = float(case["bin_width_da"])
    sigma = float(sigma_da if sigma_da is not None else MODE_SIGMA_DA[case["detector_mode"]])
    if sigma <= 0:
        raise ValueError("sigma_da must be positive")
    composition = case["atomic_fraction"]
    charge2 = case["charge2_fraction"]
    total = np.zeros_like(axis)
    n_detected = float(case["ions_evaporated"]) * float(case["detector_efficiency"])
    for element, fraction in composition.items():
        for mass, abundance in ISOTOPES[element]:
            for charge, charge_probability in ((1, 1.0 - charge2[element]), (2, charge2[element])):
                center = float(mass) / charge
                z = (axis - center) / sigma
                total += (n_detected * float(fraction) * float(abundance) * charge_probability
                          * np.exp(-0.5 * z * z) * step / (sigma * np.sqrt(2.0 * np.pi)))
    total += float(case["background_counts_per_bin"])
    return total

