"""
zarr_adapter.py

Minimal Zarr loader adapter (placeholder).
Requires: zarr (optional dependency).
"""

from __future__ import annotations
from typing import Any, Dict

try:
    import zarr
except Exception:  # pragma: no cover
    zarr = None

from .template_adapter import DatasetAdapter


class ZarrAdapter(DatasetAdapter):
    ADAPTER_ID = "zarr"

    def can_handle(self, path: str) -> bool:
        return path.lower().endswith(".zarr")

    def load(self, path: str) -> Dict[str, Any]:
        if zarr is None:
            raise RuntimeError("zarr is not installed; install pwm-core[io].")
        root = zarr.open(path, mode="r")
        # convention: "y" dataset exists
        y = root["y"][:] if "y" in root else root[:]
        meta = {"format": "zarr", "keys": list(root.array_keys()) if hasattr(root, "array_keys") else []}
        return {"y": y, "x": root["x"][:] if "x" in root else None, "meta": meta}
