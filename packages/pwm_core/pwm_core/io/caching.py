"""pwm_core.io.caching

Optional local cache for referenced datasets/operators.
Keep minimal for now; can be expanded (content-addressed cache).
"""

from __future__ import annotations

import os
import shutil
from typing import Optional


def ensure_cached(path: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, os.path.basename(path))
    if not os.path.exists(dst):
        shutil.copy2(path, dst)
    return dst
