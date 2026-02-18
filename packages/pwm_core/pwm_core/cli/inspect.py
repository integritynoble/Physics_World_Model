"""pwm_core.cli.inspect

Inspect a RunBundle: show manifest, spec, artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def inspect_runbundle(runbundle_dir: str) -> Dict[str, Any]:
    rb = Path(runbundle_dir)
    m = json.loads((rb / "runbundle_manifest.json").read_text())
    out: Dict[str, Any] = {
        "run_id": m.get("run_id"),
        "spec_version": m.get("spec_version"),
        "artifacts": m.get("artifacts", {}),
        "data": m.get("data", {}),
    }
    return out
