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


_gallery_cache: dict | None = None
_gallery_cache_ts: float = 0.0
_GALLERY_TTL: float = 3600.0  # 1 hour


def _fetch_gallery_from_gcs() -> dict | None:
    """Fetch benchmark_gallery.json from GCS using authenticated access."""
    from pwm_platform.services.gcs_signer import fetch_gcs_json

    return fetch_gcs_json("benchmark_gallery/benchmark_gallery.json")


def _load_gallery_local() -> dict | None:
    """Fallback: load gallery JSON from local static file."""
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
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_benchmark_gallery(variant_key: str) -> dict | None:
    """Load pre-computed benchmark gallery data for a variant.

    Fetches from GCS with in-memory caching (1-hour TTL).
    Falls back to local static file if GCS is unreachable.
    """
    import time

    global _gallery_cache, _gallery_cache_ts

    now = time.monotonic()
    if _gallery_cache is None or (now - _gallery_cache_ts) > _GALLERY_TTL:
        data = _fetch_gallery_from_gcs()
        if data is None:
            data = _load_gallery_local()
        if data is not None:
            _gallery_cache = data
            _gallery_cache_ts = now

    if _gallery_cache is None:
        return None
    return _gallery_cache.get(variant_key)
