"""pwm_AI_Scientist.operator_bridge

Maps AI_Scientist inputs (y, A) into PWM endpoints for operator-fit and calibration+reconstruction.

This module is optional; if AI_Scientist already stores y/A on disk, you can call pwm_fit_operator directly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pwm_core.api import endpoints


def fit_operator_from_paths(y_path: str, operator_id: str, out_dir: str = "runs") -> Dict[str, Any]:
    return endpoints.fit_operator(y_path=y_path, operator_id=operator_id, out_dir=out_dir)


def calibrate_recon_from_paths(y_path: str, operator_id: str, out_dir: str = "runs") -> Dict[str, Any]:
    return endpoints.calibrate_recon(y_path=y_path, operator_id=operator_id, out_dir=out_dir)
