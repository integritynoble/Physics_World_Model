"""
nifti_adapter.py

Minimal NIfTI loader adapter (placeholder).

Requires: nibabel (optional dependency).
"""

from __future__ import annotations
from typing import Any, Dict

try:
    import nibabel as nib
except Exception:  # pragma: no cover
    nib = None

from .template_adapter import DatasetAdapter


class NIfTI_Adapter(DatasetAdapter):
    ADAPTER_ID = "nifti"

    def can_handle(self, path: str) -> bool:
        return path.lower().endswith((".nii", ".nii.gz"))

    def load(self, path: str) -> Dict[str, Any]:
        if nib is None:
            raise RuntimeError("nibabel is not installed; install pwm-core[io].")
        img = nib.load(path)
        arr = img.get_fdata()
        meta = {"format": "nifti", "affine": img.affine.tolist()}
        return {"y": arr, "x": None, "meta": meta}
