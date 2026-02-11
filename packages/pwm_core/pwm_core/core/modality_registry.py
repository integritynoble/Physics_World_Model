"""Modality registry loader -- programmatic access to modalities.yaml."""
from __future__ import annotations
import os
from typing import Any, Dict, List, Literal, Optional
import yaml
from pydantic import BaseModel, ConfigDict


class ModalityInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")  # modalities.yaml has extra fields
    display_name: str
    category: str
    keywords: List[str] = []
    default_solver: str = "tv_fista"
    default_template_id: str = ""
    requires_x_interaction: bool = False
    acceptance_tier: str = "C"  # A, B, or C


_CACHE: Optional[Dict[str, ModalityInfo]] = None


def _modalities_path() -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "contrib", "modalities.yaml"
    )


def load_modalities() -> Dict[str, ModalityInfo]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = _modalities_path()
    with open(path) as f:
        data = yaml.safe_load(f)
    mods = {}
    for key, info in data.get("modalities", {}).items():
        mods[key] = ModalityInfo.model_validate(info)
    _CACHE = mods
    return mods


def get_modality(key: str) -> ModalityInfo:
    mods = load_modalities()
    if key not in mods:
        raise KeyError(f"Unknown modality '{key}'. Available: {sorted(mods.keys())}")
    return mods[key]


def clear_cache() -> None:
    global _CACHE
    _CACHE = None
