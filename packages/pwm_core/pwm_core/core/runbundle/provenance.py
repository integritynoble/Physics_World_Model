"""pwm_core.core.runbundle.provenance

Capture provenance for reproducibility.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def capture_provenance(rb_dir: str) -> str:
    prov: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_hash": _git_hash(),
    }
    path = os.path.join(rb_dir, "provenance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    return path
