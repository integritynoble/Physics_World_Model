"""Modality and algorithm database registry.

Loads YAML files from database/modalities/ and database/algorithms/.
Falls back to the existing platform modality_database.py for additional info.
"""
from __future__ import annotations
import os
import sys
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Any

# Resolve paths relative to this file
_DB_DIR   = Path(__file__).parent
_MOD_DIR  = _DB_DIR / "modalities"
_ALG_DIR  = _DB_DIR / "algorithms"

# ── YAML loaders ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _load_all_modalities() -> dict[str, dict]:
    """Load all YAML files in database/modalities/."""
    result: dict[str, dict] = {}
    for path in sorted(_MOD_DIR.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        if data and "name" in data:
            result[data["name"]] = data
            # Also index by display_name slug
            for alias in data.get("aliases", []):
                result[alias] = data
    return result


@lru_cache(maxsize=None)
def _load_all_algorithms() -> dict[str, dict]:
    """Load all YAML files in database/algorithms/."""
    result: dict[str, dict] = {}
    for path in sorted(_ALG_DIR.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        if data and "algorithms" in data:
            for alg in data["algorithms"]:
                result[alg["id"]] = alg
    return result


# ── Optional platform DB bridge ───────────────────────────────────────────────

@lru_cache(maxsize=None)
def _load_platform_db() -> dict[str, dict]:
    """Try to import the existing platform MODALITY_DATABASE."""
    platform_path = Path(__file__).parents[3] / "platform" / "pwm_platform" / "services"
    if str(platform_path) not in sys.path:
        sys.path.insert(0, str(platform_path))
    try:
        from modality_database import MODALITY_DATABASE  # type: ignore
        return MODALITY_DATABASE  # type: ignore[return-value]
    except ImportError:
        return {}


# ── Public API ────────────────────────────────────────────────────────────────

def get_modality(name: str) -> dict[str, Any] | None:
    """Return full modality config.  YAML-db first, then platform fallback."""
    mods = _load_all_modalities()
    if name in mods:
        return mods[name]
    # fuzzy: first modality whose name contains `name`
    for key, val in mods.items():
        if name.lower() in key.lower():
            return val
    # platform fallback
    return _load_platform_db().get(name)


def get_algorithms(modality_name: str) -> list[dict[str, Any]]:
    """Return all algorithms for a given modality (in order of complexity)."""
    mod = get_modality(modality_name)
    if not mod:
        return []
    alg_ids: list[str] = mod.get("algorithms", [])
    all_algs = _load_all_algorithms()
    result = []
    for alg_id in alg_ids:
        if alg_id in all_algs:
            result.append(all_algs[alg_id])
    return result


def get_algorithm(alg_id: str) -> dict[str, Any] | None:
    return _load_all_algorithms().get(alg_id)


def list_modalities() -> list[str]:
    return sorted(_load_all_modalities().keys())


def get_modality_context_for_prompt(name: str) -> str:
    """Return a concise text summary of the modality for injection into prompts."""
    mod = get_modality(name)
    if not mod:
        return f"(No database entry found for modality '{name}')"

    lines = [
        f"**Modality**: {mod.get('display_name', name)}",
        f"**Category**: {mod.get('category', '?')}",
        f"**Physics class**: {mod.get('physics_class', '?')}",
    ]

    if "forward_elements" in mod:
        lines.append("\n**Forward model elements**:")
        for el in mod["forward_elements"]:
            noise_types = [n.get("type", "") for n in el.get("noise", [])]
            mm_types    = [m.get("type", "") for m in el.get("mismatch", [])]
            lines.append(
                f"  - `{el['id']}` ({el['type']})"
                + (f" noise={noise_types}" if noise_types else "")
                + (f" mismatch={mm_types}" if mm_types else "")
            )

    if "algorithms" in mod:
        lines.append(f"\n**Algorithms**: {', '.join(mod['algorithms'])}")

    if "mismatch_sources" in mod:
        lines.append(f"**Known mismatch sources**: {', '.join(mod['mismatch_sources'])}")

    if "budget_estimate" in mod:
        b = mod["budget_estimate"]
        lines.append(
            f"**Budget estimate**: equipment ~${b.get('equipment_k_usd','?')}k, "
            f"per-scan ~${b.get('per_scan_usd','?')}"
        )

    return "\n".join(lines)


def get_algorithm_context_for_prompt(modality_name: str) -> str:
    """Return algorithm catalog text for injection into prompts."""
    algs = get_algorithms(modality_name)
    if not algs:
        return f"(No algorithm entries found for modality '{modality_name}')"
    lines = ["**Available algorithms** (id / name / type / reference):"]
    for alg in algs:
        lines.append(
            f"  - `{alg['id']}`: {alg.get('name',alg['id'])} "
            f"[{alg.get('type','?')}] — {alg.get('reference','')}"
        )
    return "\n".join(lines)
