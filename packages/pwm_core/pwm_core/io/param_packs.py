"""pwm_core.io.param_packs

Parameter packs are versioned JSON bundles that provide realistic defaults
for sensors/objectives/illumination etc.

Contrib packs live in contrib/param_packs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_param_pack(pack_path: str) -> Dict[str, Any]:
    return json.loads(Path(pack_path).read_text(encoding="utf-8"))
