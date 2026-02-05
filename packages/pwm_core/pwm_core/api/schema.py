"""
pwm_core.api.schema

JSON Schema generation + version guardrails.

We export:
- ExperimentSpec v0.2.1
- DiagnosisResult + Action
- RunBundle manifests (see docs/runbundle_format.md)

This module keeps schema generation *pure* (no file IO by default).
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel

from .types import ExperimentSpec, DiagnosisResult, Action, CalibReconResult


def _schema_for(model: type[BaseModel]) -> Dict[str, Any]:
    # Pydantic v2: model.model_json_schema()
    return model.model_json_schema()


def get_schemas() -> Dict[str, Dict[str, Any]]:
    return {
        "ExperimentSpec_v0.2.1": _schema_for(ExperimentSpec),
        "DiagnosisResult_v0.2.1": _schema_for(DiagnosisResult),
        "Action_v0.2.1": _schema_for(Action),
        "CalibReconResult_v0.2.1": _schema_for(CalibReconResult),
    }


def write_schemas(out_dir: str) -> None:
    """Optional helper: write schema JSONs to a directory."""
    import os, json
    os.makedirs(out_dir, exist_ok=True)
    schemas = get_schemas()
    for k, v in schemas.items():
        path = os.path.join(out_dir, f"{k}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(v, f, indent=2)
