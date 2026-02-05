"""pwm_core.core.runbundle.manifest

Helpers for writing data_manifest.json.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict


def write_data_manifest(rb_dir: str, manifest: Dict[str, Any]) -> str:
    path = os.path.join(rb_dir, "data_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
