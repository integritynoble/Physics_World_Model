"""
Benchmark Database — static data for modality variant benchmark pages.

Contains spec primitives, variant-specific benchmark configurations (Benchmark 1-4),
leaderboard data sourced from the InverseNet paper, flowcharts, and credits
configuration. Pattern follows modality_database.py (pure data, no DB models).

Public API (unchanged from the original monolithic module):
    get_variant(variant_key)        -> dict | None
    list_all_variant_keys()         -> list[str]
    list_variants_for_modality(key) -> list[str]
    get_flowcharts()                -> dict
    get_spec_primitives()           -> dict
    VARIANT_DATABASE                -> dict[str, dict]
"""

from __future__ import annotations

from ._factory import build_variant
from ._flowcharts import FLOWCHARTS
from ._leaderboard_data import LEADERBOARD_DATA
from ._primitives import SPEC_PRIMITIVES
from ._variant_registry import VARIANT_REGISTRY

# ── Build the full database at import time ────────────────────────────────────

VARIANT_DATABASE: dict[str, dict] = {}
for _key, _entry in VARIANT_REGISTRY.items():
    VARIANT_DATABASE[_key] = build_variant(_key, _entry, LEADERBOARD_DATA.get(_key))

# ── Auto-derive modality → variants mapping ──────────────────────────────────

_MODALITY_TO_VARIANTS: dict[str, list[str]] = {}
for _key, _entry in VARIANT_DATABASE.items():
    _parent = _entry["parent_modality"]
    _MODALITY_TO_VARIANTS.setdefault(_parent, []).append(_key)


# ── Public API ────────────────────────────────────────────────────────────────


def get_variant(variant_key: str) -> dict | None:
    """Return full variant record, or None if not found."""
    entry = VARIANT_DATABASE.get(variant_key)
    if entry is None:
        return None
    return dict(entry)


def list_all_variant_keys() -> list[str]:
    """Return all variant keys in insertion order."""
    return list(VARIANT_DATABASE.keys())


def list_variants_for_modality(modality_key: str) -> list[str]:
    """Return variant keys belonging to a parent modality (e.g. 'spc' -> ['spc_block', 'spc_kronecker'])."""
    return list(_MODALITY_TO_VARIANTS.get(modality_key, []))


def get_flowcharts() -> dict:
    """Return the flowchart definitions."""
    return dict(FLOWCHARTS)


def get_spec_primitives() -> dict:
    """Return the 11 spec primitives."""
    return dict(SPEC_PRIMITIVES)


def get_benchmark_gallery(variant_key: str) -> dict | None:
    """Load pre-computed benchmark gallery data for a variant.

    Returns the gallery dict for the variant from benchmark_gallery.json,
    or None if the file doesn't exist or the variant has no gallery data.
    """
    import json
    from pathlib import Path

    json_path = (
        Path(__file__).resolve().parent.parent.parent
        / "static" / "benchmark-data" / "benchmark_gallery.json"
    )
    if not json_path.exists():
        return None
    try:
        with open(json_path) as f:
            gallery = json.load(f)
        return gallery.get(variant_key)
    except (json.JSONDecodeError, OSError):
        return None
