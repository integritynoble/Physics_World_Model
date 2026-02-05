"""
ome_tiff_adapter.py

Minimal OME-TIFF loader adapter (placeholder).

Requires: tifffile (optional dependency).
If tifffile isn't installed, this adapter should not be registered.
"""

from __future__ import annotations
from typing import Any, Dict

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None

from .template_adapter import DatasetAdapter


class OME_TIFF_Adapter(DatasetAdapter):
    ADAPTER_ID = "ome_tiff"

    def can_handle(self, path: str) -> bool:
        return path.lower().endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff"))

    def load(self, path: str) -> Dict[str, Any]:
        if tifffile is None:
            raise RuntimeError("tifffile is not installed; install pwm-core[io].")
        arr = tifffile.imread(path)
        meta = {"format": "ome-tiff", "notes": "OME metadata parsing can be added here."}
        # Heuristic: treat as y unless you provide a manifest
        return {"y": arr, "x": None, "meta": meta}
