"""pwm_core.physics.spectral.dispersion_models

Dispersion parameterizations for spectral operators (CASSI, etc.).

Provides:
- polynomial dispersion: dx(l) = a0 + a1*l + a2*l^2 ...
- LUT dispersion
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


def dispersion_shift(theta: Dict[str, Any], band: int) -> Tuple[float, float]:
    """Return (dx, dy) for spectral band index."""
    model = theta.get("dispersion_model", "poly")
    if model == "poly":
        ax = theta.get("disp_poly_x", [0.0, 1.0, 0.0])
        ay = theta.get("disp_poly_y", [0.0, 0.0, 0.0])
        l = float(band)
        dx = sum(a * (l ** i) for i, a in enumerate(ax))
        dy = sum(a * (l ** i) for i, a in enumerate(ay))
        return dx, dy
    if model == "lut":
        lut = theta.get("disp_lut", {})
        dx = float(lut.get(str(band), {}).get("dx", 0.0))
        dy = float(lut.get(str(band), {}).get("dy", 0.0))
        return dx, dy
    return 0.0, 0.0
