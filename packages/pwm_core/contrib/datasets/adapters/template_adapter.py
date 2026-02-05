"""
template_adapter.py

Contributor template for a dataset adapter.

A dataset adapter turns a file/folder into a standard dict:
{
  "y": torch.Tensor or np.ndarray,
  "x": optional torch.Tensor or np.ndarray,
  "meta": {
      "modality": "...",
      "axes": "HWC/L", ...
      "units": {...},
      "notes": "...",
  }
}
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class DatasetAdapter:
    ADAPTER_ID: str = "template"

    def can_handle(self, path: str) -> bool:
        raise NotImplementedError

    def load(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError


class TemplateAdapter(DatasetAdapter):
    ADAPTER_ID = "template"

    def can_handle(self, path: str) -> bool:
        return path.endswith(".npz")

    def load(self, path: str) -> Dict[str, Any]:
        data = np.load(path)
        y = data["y"]
        x = data["x"] if "x" in data else None
        meta = dict(data["meta"].item()) if "meta" in data else {}
        return {"y": y, "x": x, "meta": meta}
