"""The PWM imaging catalog: every registered imaging modality as one design record.

Each entry is a spec.md design record with the eight fields of the system-design paper (modality, carrier, geometry,
object, forward_model, noise, target, system_elements), plus its status in each registry and an honest validation
level. Sources:

  - pwm_core contrib/modalities.yaml   170 modalities with element chains (source, optics, detector, ...)
  - pwm_core contrib/graph_templates.yaml + GraphCompiler   does a forward model compile and run?
  - pwm_core canonical_decompositions  31 validated canonical chains (Finite Primitive Basis paper, Table 1)
  - genesis l1/l2/l3                   500 principles; the imaging ones are linked by name (a candidate link)
  - the flagship paper's 12 modalities with hardware (real-data) validation

Validation levels (lowest to highest): registered < runs (a template compiles and runs) < canonical (its chain is in
the validated registry) < real_data (hardware validation in the flagship paper). A level is never inferred from a
name alone.

Build once with `python -m pwm_mode.imaging.catalog --build` (writes catalog.json next to this file).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import core

CATALOG_PATH = Path(__file__).with_name("catalog.json")

# Flagship paper, "Empirical Validation": hardware validation on real data, all five carrier families.
REAL_DATA = {"cassi", "cacti", "spc", "lensless", "compressive_holography", "fluorescence_microscopy", "ct", "cbct",
             "electron_ptychography", "cryo_em", "mri", "ultrasound"}
IMAGING_FAMILIES = ("optical", "acoustic", "xray", "electron", "nuclear_spin")
CARRIER_WORDS = [("xray", ("x-ray", "xray", "ct", "radiograph", "mammo", "fluoroscop", "tomosynth")),
                 ("electron", ("electron", "tem", "sem", "stem", "cryo")),
                 ("acoustic", ("ultrasound", "acoustic", "sonar", "seismic", "photoacoustic", "elastograph")),
                 ("spin", ("mri", "magnetic resonance", "nmr", "epr", "spin")),
                 ("particle", ("neutron", "muon", "proton", "pet", "spect", "gamma", "nuclear")),
                 ("photon", ("optical", "fluoresc", "microscop", "camera", "lidar", "spectral", "laser", "light"))]
NOISE_ELEMENTS = {"shot_poisson": "Poisson (shot)", "read_gaussian": "Gaussian (read)"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def _carrier(m: Dict[str, Any], principle: Optional[Dict[str, Any]]) -> str:
    if principle and principle.get("carrier"):
        return str(principle["carrier"]).lower().replace("-", "")
    text = " ".join([m.get("display_name", ""), m.get("category", ""), " ".join(m.get("keywords", []))]).lower()
    for carrier, words in CARRIER_WORDS:
        if any(w in text for w in words):
            return carrier
    return "unknown"


class _Timeout(Exception):
    pass


def _run(template: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    nodes, notes = core.normalise_nodes(template["nodes"])
    signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Timeout()))
    try:
        signal.alarm(15)
        op = core.compile_graph(nodes, template["edges"], template.get("metadata", {}))
        x = np.random.default_rng(0).random(tuple(template.get("metadata", {}).get("x_shape", [64, 64])))
        y = op.forward(x)
        ok = bool(np.all(np.isfinite(np.asarray(y, dtype=complex))))
        out = {"runs": ok, "canonical_chain": op.canonical_dag_string(), "aliases_applied": notes,
               "nodes": len(op.forward_plan)}
        if template.get("metadata", {}).get("repaired_by"):
            out["repaired_by"] = template["metadata"]["repaired_by"]
        return out
    except _Timeout:
        return {"runs": False, "error": "timeout", "aliases_applied": notes}
    except Exception as e:  # noqa: BLE001 - the error text is the record
        return {"runs": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "aliases_applied": notes}
    finally:
        signal.alarm(0)


def _system_elements(m: Dict[str, Any], principle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in m.get("elements", []):
        groups.setdefault(e.get("element_type", "other"), []).append(
            {"name": e.get("name"), "transfer": e.get("transfer_kind"), "throughput": e.get("throughput"),
             "parameters": e.get("parameters", {}), "noise": e.get("noise_kinds", [])})
    calib = (principle or {}).get("mismatch_parameters") or []
    return {"source": groups.pop("source", []), "detector": groups.pop("detector", []),
            "optics_and_other": groups, "calibration": [{"parameter": p, "tolerance": None} for p in calib]}


def build() -> Dict[str, Any]:
    import glob
    mods = core.contrib("modalities.yaml")["modalities"]
    templates = core.contrib("graph_templates.yaml")["templates"]
    canon = core.canonical_decompositions()
    genesis = {}
    for f in glob.glob(str(core.GENESIS_SRC / "l1" / "*.json")):
        p = json.load(open(f))
        if p.get("carrier_family") in IMAGING_FAMILIES:
            genesis[p["artifact_id"]] = p
    titles = {k: _norm(v["title"]) for k, v in genesis.items()}
    entries = []
    linked = set()
    for mid, m in mods.items():
        tids = [k for k, t in templates.items() if t.get("metadata", {}).get("modality") == mid]
        runs = {k: _run(templates[k]) for k in tids}
        best = difflib.get_close_matches(_norm(m.get("display_name", mid)), list(titles.values()), n=1, cutoff=0.6)
        gid = next((k for k, v in titles.items() if best and v == best[0]), None)
        principle = genesis.get(gid) if gid else None
        if gid:
            linked.add(gid)
        running = [k for k, r in runs.items() if r.get("runs")]
        chain = runs[running[0]]["canonical_chain"] if running else None
        level = "registered"
        if running:
            level = "runs"
        if mid in canon:
            level = "canonical"
        if mid in REAL_DATA:
            level = "real_data"
        noise = sorted({n for e in m.get("elements", []) for n in e.get("noise_kinds", [])})
        entries.append({
            "id": mid, "title": m.get("display_name", mid), "category": m.get("category"),
            "keywords": m.get("keywords", []), "summary": " ".join((m.get("description") or "").split())[:400],
            "spec_md": {
                "modality": mid,
                "carrier": _carrier(m, principle),
                "geometry": {k: v for k, v in m.items() if k in ("signal_dims", "wavelength_range_nm")},
                "object": (principle or {}).get("world_state_x") or {"x": (m.get("signal_dims") or {}).get("x")},
                "forward_model": {"equation": m.get("forward_model_equation"), "type": m.get("forward_model_type"),
                                  "canonical_chain": chain, "validated_chain": canon.get(mid),
                                  "graph_templates": tids},
                "noise": noise,
                "target": {"metric": (principle or {}).get("error_metric") or "PSNR_dB",
                           "secondary": (principle or {}).get("error_metric_secondary")},
                "system_elements": _system_elements(m, principle),
            },
            "default_solver": m.get("default_solver"),
            "upload_template": m.get("upload_template"),
            "templates": runs,
            "genesis_principle": gid, "genesis_title": (principle or {}).get("title"),
            "level": level,
            "gaps": [g for g, bad in (("no running template", not running),
                                      ("no source element (source design not expressible)",
                                       not any(e.get("element_type") == "source" for e in m.get("elements", []))),
                                      ("no linked genesis principle", gid is None),
                                      ("calibration tolerances not recorded", True)) if bad],
        })
    principle_only = [{"id": k, "title": v["title"], "carrier": v.get("carrier"), "dag": v.get("dag"),
                       "level": "principle_only", "gaps": ["not in the modality registry: curate (imaging or not?)"]}
                      for k, v in sorted(genesis.items()) if k not in linked]
    cat = {"version": 1, "modalities": entries, "principle_only": principle_only,
           "summary": {"registry_modalities": len(entries),
                       "runs": sum(e["level"] != "registered" for e in entries),
                       "canonical": sum(e["level"] in ("canonical", "real_data") for e in entries),
                       "real_data": sum(e["level"] == "real_data" for e in entries),
                       "with_source_element": sum("no source element (source design not expressible)" not in e["gaps"]
                                                  for e in entries),
                       "genesis_linked": sum(bool(e["genesis_principle"]) for e in entries),
                       "principle_only": len(principle_only)}}
    return cat


def load() -> Dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text())


def search(query: str = "", carrier: Optional[str] = None, category: Optional[str] = None,
           min_level: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    order = ["registered", "runs", "canonical", "real_data"]
    words = [w for w in _norm(query).split() if w]
    out = []
    for e in load()["modalities"]:
        if carrier and e["spec_md"]["carrier"] != carrier.lower():
            continue
        if category and (e.get("category") or "").lower() != category.lower():
            continue
        if min_level and order.index(e["level"]) < order.index(min_level):
            continue
        text = _norm(" ".join([e["id"], e["title"], e.get("category") or "", " ".join(e["keywords"]), e["summary"]]))
        hits = sum(w in text for w in words)
        if words and not hits:
            continue
        out.append({"id": e["id"], "title": e["title"], "carrier": e["spec_md"]["carrier"], "level": e["level"],
                    "chain": e["spec_md"]["forward_model"]["canonical_chain"], "gaps": e["gaps"], "hits": hits})
    out.sort(key=lambda r: (-r["hits"], -order.index(r["level"]), r["id"]))
    return out[:limit]


def get(modality: str) -> Dict[str, Any]:
    for e in load()["modalities"]:
        if e["id"] == modality:
            return e
    raise KeyError(f"no modality {modality!r} in the imaging catalog")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        cat = build()
        CATALOG_PATH.write_text(json.dumps(cat, indent=1, default=str))
        print(json.dumps(cat["summary"], indent=1))
