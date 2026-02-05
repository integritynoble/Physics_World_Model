"""pwm_core.core.prompt_compiler

Rule-based PromptCompiler:
prompt -> (casepack_id, draft_spec, assumptions, patch_list)

Deterministic and safe: selects a CasePack and applies *stated* overrides.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class CompileResult:
    casepack_id: str
    draft_spec: Dict[str, Any]
    assumptions: List[str]
    patch_list: List[Dict[str, Any]]


def _score_pack(prompt: str, pack: Dict[str, Any]) -> int:
    p = prompt.lower()
    score = 0
    modality = pack.get("states", {}).get("physics", {}).get("modality", "")
    if modality and modality.lower() in p:
        score += 5
    for kw in pack.get("keywords", []):
        if kw.lower() in p:
            score += 1
    for t in pack.get("tags", []):
        if t.lower() in p:
            score += 1
    return score


def select_casepack(prompt: str, casepack_dir: str) -> Dict[str, Any]:
    best, best_score = None, -1
    for fp in Path(casepack_dir).glob("*.json"):
        pack = json.loads(fp.read_text(encoding="utf-8"))
        s = _score_pack(prompt, pack)
        if s > best_score:
            best, best_score = pack, s
    if best is None:
        raise FileNotFoundError(f"No casepacks found in {casepack_dir}")
    return best


def extract_overrides(prompt: str) -> Tuple[Dict[str, Any], List[str]]:
    p = prompt.lower()
    overrides: Dict[str, Any] = {}
    assumptions: List[str] = []

    if "low dose" in p or "low-dose" in p or "phototoxic" in p:
        overrides.setdefault("states", {}).setdefault("budget", {}).setdefault("photon_budget", {})["max_photons"] = 300.0
    if "high dose" in p:
        overrides.setdefault("states", {}).setdefault("budget", {}).setdefault("photon_budget", {})["max_photons"] = 3000.0

    m = re.search(r"sampling\s*rate\s*=?\s*([0-9]*\.?[0-9]+)", p)
    if m:
        overrides.setdefault("states", {}).setdefault("budget", {}).setdefault("measurement_budget", {})["sampling_rate"] = float(m.group(1))

    m2 = re.search(r"(\d+)\s*(frames|views|patterns)", p)
    if m2:
        overrides.setdefault("states", {}).setdefault("budget", {}).setdefault("measurement_budget", {})["num_measurements"] = int(m2.group(1))

    if not overrides:
        assumptions.append("No explicit numeric overrides found; using CasePack defaults.")
    return overrides, assumptions


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def compile_prompt_to_spec(prompt: str, casepack_dir: str) -> CompileResult:
    pack = select_casepack(prompt, casepack_dir)
    overrides, assumptions = extract_overrides(prompt)
    draft = deep_merge(pack.get("base_spec", {}), overrides)

    patch_list = []
    if overrides:
        patch_list.append({"op": "merge", "value": overrides})

    return CompileResult(
        casepack_id=pack.get("id", "unknown_casepack"),
        draft_spec=draft,
        assumptions=assumptions,
        patch_list=patch_list,
    )
