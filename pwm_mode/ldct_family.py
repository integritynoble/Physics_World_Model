"""A family of low-dose CT problems from the real paired corpus, for the knowledge engine.

The runtime's benchmark has one problem per patient (the middle slice, a 150 HU, 5 px lesion). The engine needs many
problems that share conditions across patients, so that a setting tuned on one patient can be tried on another:

  problem = (patient, slice, lesion contrast in HU, lesion radius in px)

The lesion is inserted the runtime's way (its own ``pick_site`` and ``insert``, imported, not copied), into both the
low-dose input and the full-dose answer key at the same place. The answer key, the mask and the amplitude go to the
protected store; the solver's workspace gets the low-dose slice and the problem's public conditions only.

Splits are by patient: a patient is either for development (its problems may build the library) or held out (its
problems are only ever evaluated). That keeps a patient's anatomy and noise texture from leaking from library to test.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

RUNTIME_SRC = os.environ.get("AI4SCIENCE_SRC", "/home/spiritai/pwm/Physics_World_Model/AI4Science")
DATA = Path(os.environ.get("PWM_LDCT_CORPUS") or
            Path(os.environ.get("AI4SCIENCE_DATA", "/home/spiritai/.ai4science/data")) / "ldct")
_GEN = Path(RUNTIME_SRC) / "ai4science/harness/agents/research_agents/runners/payload/ldct/generate.py"

CONTRASTS = (100.0, 150.0, 200.0)      # HU; faint to clear against ~100 HU full-dose noise
RADII = (3, 5, 7)                      # px


def _generator():
    spec = importlib.util.spec_from_file_location("ldct_generate", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def corpus():
    meta = json.loads((DATA / "metadata.json").read_text())
    vols = np.load(DATA / "volumes.npz")
    return sorted(meta["patients"]), vols


def anatomy(patient: str) -> str:
    return {"C": "chest", "L": "abdomen", "N": "head"}.get(patient[:1], "other")


def split(patients: List[str], held_out_every: int = 2) -> Dict[str, List[str]]:
    """Deterministic patient-level split: by a hash of the patient id, never by position."""
    ranked = sorted(patients, key=lambda p: hashlib.sha256(p.encode()).hexdigest())
    test = ranked[::held_out_every]
    return {"development": [p for p in ranked if p not in test], "held_out": test}


def problem_id(patient: str, slice_: int, hu: float, r: int) -> str:
    return f"ldct-{patient.lower()}-s{slice_}-c{int(hu)}-r{r}"


def conditions(patient: str, hu: float, r: int) -> Dict[str, Any]:
    """What a solver may know about a problem before solving it (no key-derived values)."""
    return {"anatomy": anatomy(patient), "lesion_hu": float(hu), "lesion_r": int(r)}


def make(store: Path, patient: str, slice_: int, hu: float, r: int) -> Dict[str, Any]:
    """Write one problem: the key into ``store/keys/<pid>``, the public part into ``store/public/<pid>``."""
    gen = _generator()
    _, vols = corpus()
    full = vols[f"full_{patient}"][slice_].astype(np.float32)
    low = vols[f"low_{patient}"][slice_].astype(np.float32)
    cy, cx = gen.pick_site(full, r)
    full_l, mask = gen.insert(full, cy, cx, r, hu)
    low_l, _ = gen.insert(low, cy, cx, r, hu)
    pid = problem_id(patient, slice_, hu, r)
    key, pub = store / "keys" / pid, store / "public" / pid
    key.mkdir(parents=True, exist_ok=True)
    (pub / "data").mkdir(parents=True, exist_ok=True)
    np.save(key / "full_dose.npy", full_l)
    np.save(key / "lesion_mask.npy", mask)
    np.save(key / "lesion_amplitude.npy", np.array([hu, r], dtype=np.float32))
    np.save(key / "low_dose.npy", low_l)
    np.save(pub / "data" / "low_dose.npy", low_l)
    spec = {"problem_id": pid, "patient": patient, "slice": slice_, "conditions": conditions(patient, hu, r),
            "question": "restore the low-dose slice without losing the inserted low-contrast lesion",
            "deliverable": "results/reconstruction.npy, same shape as data/low_dose.npy, in HU",
            "criteria": ["PSNR above the untouched low-dose slice", "lesion CNR >= 3 (Rose criterion)",
                         "at least half of the inserted contrast retained"]}
    (pub / "problem.json").write_text(json.dumps(spec, indent=1))
    (key / "problem.json").write_text(json.dumps(spec, indent=1))
    return spec


def build_family(store: Path, patients: List[str] | None = None, slices=(0, 1, 2),
                 contrasts=CONTRASTS, radii=RADII) -> List[Dict[str, Any]]:
    pats, vols = corpus()
    out = []
    for p in patients or pats:
        n = vols[f"full_{p}"].shape[0]
        for s in slices:
            if s >= n:
                continue
            for hu in contrasts:
                for r in radii:
                    out.append(make(Path(store), p, s, hu, r))
    return out
