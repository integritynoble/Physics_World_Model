"""pwm_AI_Scientist.tools

Thin wrappers that expose pwm_core.api.endpoints as AI_Scientist-callable tools.

This file intentionally has no AI_Scientist imports to keep it reusable;
AI_Scientist integration can call these functions directly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pwm_core.api import endpoints


def pwm_run(prompt: Optional[str] = None, spec: Optional[Dict[str, Any]] = None, out_dir: str = "runs") -> Dict[str, Any]:
    return endpoints.run(prompt=prompt, spec=spec, out_dir=out_dir)


def pwm_fit_operator(y_path: str, operator_id: str, out_dir: str = "runs") -> Dict[str, Any]:
    return endpoints.fit_operator(y_path=y_path, operator_id=operator_id, out_dir=out_dir)


def pwm_calibrate_recon(y_path: str, operator_id: str, out_dir: str = "runs") -> Dict[str, Any]:
    return endpoints.calibrate_recon(y_path=y_path, operator_id=operator_id, out_dir=out_dir)
