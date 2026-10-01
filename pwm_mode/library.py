"""The PWM library as PWM mode sees it (test side): versioned states of principles (L1), digital twins (L2),
benchmarks (L3) and accepted solutions (L4).

The engine this exists for: PWM mode solves, an accepted solution enters the library, and the next problem starts
from it. So the library is read-only to a run (pinned to one state for the whole run), and it grows only through
``accept``, which needs a passing result from a validator that is not the submitter. Every state has a digest over its
artifacts, so a run can say exactly what it read and a held-out set can be frozen after a known state.

Layout under ``root``: ``artifacts/<id>@<v>.json``, ``states/<n>.json`` (n = 0, 1, ...), ``submissions/<sid>.json``.
Plain JSON files, written atomically; one writer at a time (a lock file).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

LAYERS = {"L1": "principle", "L2": "digital twin", "L3": "benchmark", "L4": "solution"}
_ID = re.compile(r"[a-z0-9][a-z0-9-]{1,80}")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(obj) -> str:
    return hashlib.sha256(_canon(obj)).hexdigest()


class LibraryError(ValueError):
    pass


class Library:
    def __init__(self, root):
        self.root = Path(root)
        for d in ("artifacts", "states", "submissions"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    # ---- storage ----
    @contextmanager
    def _locked(self):
        with open(self.root / ".lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _write(self, path: Path, obj) -> None:
        tmp = path.with_name(path.name + "." + secrets.token_hex(4) + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _read(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    # ---- states ----
    def head(self) -> int:
        ns = [int(p.stem) for p in (self.root / "states").glob("*.json") if p.stem.isdigit()]
        return max(ns) if ns else -1

    def state(self, n: Optional[int] = None) -> Dict[str, Any]:
        n = self.head() if n is None else int(n)
        p = self.root / "states" / f"{n}.json"
        if n < 0 or not p.exists():
            raise LibraryError(f"no library state {n}")
        return self._read(p)

    def _artifact(self, ref: str) -> Dict[str, Any]:
        p = self.root / "artifacts" / f"{ref}.json"
        if not p.exists():
            raise LibraryError(f"no artifact {ref}")
        return self._read(p)

    def _new_state(self, refs: List[str], parent: Optional[int], added: List[str], why: str) -> Dict[str, Any]:
        n = self.head() + 1
        content = {ref: self._artifact(ref)["digest"] for ref in sorted(refs)}
        st = {"n": n, "label": f"L{n}", "parent": parent, "artifacts": sorted(refs), "added": added,
              "digest": _sha(content), "created": _now(), "why": why}
        self._write(self.root / "states" / f"{n}.json", st)
        return st

    def _put_artifact(self, art: Dict[str, Any]) -> str:
        if art.get("layer") not in LAYERS:
            raise LibraryError("layer is one of " + ", ".join(LAYERS))
        if not _ID.fullmatch(art.get("id", "")):
            raise LibraryError("an artifact id is lowercase letters, digits and '-'")
        versions = [int(p.stem.split("@")[1]) for p in (self.root / "artifacts").glob(f"{art['id']}@*.json")]
        art = dict(art, version=(max(versions) + 1 if versions else 1))
        art["digest"] = _sha({k: v for k, v in art.items() if k not in ("digest", "created")})
        art.setdefault("created", _now())
        ref = f"{art['id']}@{art['version']}"
        self._write(self.root / "artifacts" / f"{ref}.json", art)
        return ref

    def seed(self, artifacts: Iterable[Dict[str, Any]], why: str = "seed") -> Dict[str, Any]:
        """The first state, L0: the field's principles, twins and benchmarks (no accepted solutions needed)."""
        with self._locked():
            if self.head() >= 0:
                raise LibraryError("the library already has a state; it grows only by accepting submissions")
            refs = [self._put_artifact(a) for a in artifacts]
            return self._new_state(refs, None, refs, why)

    # ---- reading (a run pins one state) ----
    def get(self, ref: str, state: Optional[int] = None) -> Dict[str, Any]:
        st = self.state(state)
        if ref not in st["artifacts"]:
            raise LibraryError(f"{ref} is not in library state L{st['n']}")
        return self._artifact(ref)

    def search(self, field: Optional[str] = None, layer: Optional[str] = None, query: str = "",
               conditions: Optional[Dict[str, Any]] = None, state: Optional[int] = None,
               limit: int = 20) -> List[Dict[str, Any]]:
        st = self.state(state)
        words = [w for w in re.split(r"\W+", (query or "").lower()) if w]
        out = []
        for ref in st["artifacts"]:
            a = self._artifact(ref)
            if field and a.get("field") != field or layer and a.get("layer") != layer:
                continue
            text = " ".join(str(a.get(k, "")) for k in ("title", "summary")).lower()
            hit = sum(1 for w in words if w in text)
            if words and not hit:
                continue
            match = condition_match(a.get("conditions") or {}, conditions or {})
            out.append({"ref": ref, "layer": a["layer"], "field": a.get("field"), "title": a.get("title"),
                        "summary": a.get("summary", ""), "conditions": a.get("conditions") or {},
                        "objective": (a.get("evidence") or {}).get("objective"), "match": match, "hits": hit})
        out.sort(key=lambda r: (-r["match"], -r["hits"], -(r["objective"] or 0.0), r["ref"]))
        return out[:limit]

    def best_solution(self, field: str, conditions: Dict[str, Any], state: Optional[int] = None):
        """The accepted solution that fits these conditions best (most matching conditions, then the best validated
        objective), or None when the library holds none for the field."""
        rows = self.search(field=field, layer="L4", conditions=conditions, state=state, limit=1000)
        rows = [r for r in rows if r["match"] > 0] or rows
        return self._artifact(rows[0]["ref"]) if rows else None

    # ---- growing (only through a validator) ----
    def submit(self, *, field: str, layer: str, id: str, title: str, summary: str, payload: Dict[str, Any],
               conditions: Dict[str, Any], source_problems: List[str], submitter: str,
               based_on: Optional[str] = None, evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if layer not in LAYERS:
            raise LibraryError("layer is one of " + ", ".join(LAYERS))
        sid = "sub-" + secrets.token_hex(6)
        sub = {"id": sid, "state": "pending", "created": _now(), "submitter": submitter,
               "artifact": {"id": id, "layer": layer, "field": field, "title": title, "summary": summary,
                            "payload": payload, "conditions": conditions, "source_problems": source_problems,
                            "based_on": based_on, "contributors": [submitter] + ([based_on] if based_on else [])},
               "claimed": evidence or {}}
        self._write(self.root / "submissions" / f"{sid}.json", sub)
        return sub

    def accept(self, sid: str, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Add a submission's artifact as a new library state. Needs the validator's own passing result; the
        submitter's claimed evidence is kept but never counts."""
        with self._locked():
            p = self.root / "submissions" / f"{sid}.json"
            if not p.exists():
                raise LibraryError(f"no submission {sid}")
            sub = self._read(p)
            if sub["state"] != "pending":
                raise LibraryError(f"submission {sid} is already {sub['state']}")
            if validation.get("passed") is not True:
                raise LibraryError("only a passing validation is accepted")
            if not validation.get("validator") or validation.get("validator") == sub["submitter"]:
                raise LibraryError("the validator must be named and must not be the submitter")
            parent = self.state()
            art = dict(sub["artifact"], evidence={k: validation[k] for k in validation
                                                  if k in ("validator", "metrics", "objective", "problems", "receipt")})
            ref = self._put_artifact(art)
            prev = [r for r in parent["artifacts"] if r.split("@")[0] != art["id"]]   # a new version replaces the old
            st = self._new_state(prev + [ref], parent["n"], [ref], f"accepted {sid}")
            sub.update(state="accepted", accepted_as=ref, library_state=st["n"], validation=validation)
            self._write(p, sub)
            return st

    def reject(self, sid: str, why: str) -> Dict[str, Any]:
        with self._locked():
            p = self.root / "submissions" / f"{sid}.json"
            sub = self._read(p)
            if sub["state"] != "pending":
                raise LibraryError(f"submission {sid} is already {sub['state']}")
            sub.update(state="rejected", why=why)
            self._write(p, sub)
            return sub


def condition_match(have: Dict[str, Any], want: Dict[str, Any]) -> int:
    """How many of the wanted conditions this artifact shares (exact for text, equal for numbers)."""
    n = 0
    for k, v in want.items():
        if k in have and (have[k] == v or (isinstance(v, (int, float)) and isinstance(have[k], (int, float))
                                           and abs(float(have[k]) - float(v)) < 1e-9)):
            n += 1
    return n
