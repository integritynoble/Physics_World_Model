"""pwm_core.export.package

Reference-aware packaging:
- if data is large, keep as reference and write data_manifest.json pointer
- otherwise copy into RunBundle and include in archive
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import os
import tarfile
import zipfile


@dataclass
class PackagePolicy:
    max_copy_bytes: int = 100 * 1024 * 1024  # 100MB default
    format: str = "zip"  # "zip" | "tar.gz"


def _dir_size_bytes(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for fn in files:
            total += (Path(root) / fn).stat().st_size
    return total


def export_runbundle(runbundle_dir: Path, out_path: Path, policy: PackagePolicy) -> Path:
    runbundle_dir = Path(runbundle_dir)
    out_path = Path(out_path)

    if policy.format == "zip":
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(runbundle_dir):
                for fn in files:
                    full = Path(root) / fn
                    rel = full.relative_to(runbundle_dir)
                    z.write(full, arcname=str(rel))
    elif policy.format == "tar.gz":
        with tarfile.open(out_path, "w:gz") as t:
            t.add(runbundle_dir, arcname=".")
    else:
        raise ValueError(f"Unknown package format: {policy.format}")
    return out_path
