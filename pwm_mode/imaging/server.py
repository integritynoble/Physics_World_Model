"""MCP server: the PWM imaging library (catalog of registered modalities + design of new ones).

Kept separate from pwm_mode.mcp_server so it can be merged without touching the family refactors on other branches;
the tools are free (no ledger), because they read registries and run deterministic checks.

    python -m pwm_mode.imaging.server        # stdio
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.mcpserver import MCPServer

from . import catalog, core, design, lifecycle

server = MCPServer(
    name="ai4science-pwm-imaging",
    version="0.1.0",
    instructions=(
        "PWM imaging library. imaging_search finds registered modalities (170, each a spec.md design record with its "
        "validation level and gaps); imaging_modality returns one record; imaging_primitives lists the building "
        "blocks; imaging_design_check compiles a new design and runs the structural compiler (C1-C6) and the three "
        "gates; imaging_design_submit files a passing design for independent validation. Every registered modality "
        "also has the four lifecycle functions: imaging_lifecycle_design, imaging_simulate, imaging_reconstruct, "
        "imaging_calibrate (arrays as .npy files). Report a check's own status: 'unassessed' is not a pass."),
)


def _dump(o) -> str:
    return json.dumps(o, indent=1, default=str)


@server.tool()
def imaging_search(query: str = "", carrier: Optional[str] = None, category: Optional[str] = None,
                   min_level: Optional[str] = None, limit: int = 20) -> str:
    """Search registered imaging modalities by words, carrier (photon, xray, electron, acoustic, spin, particle) and
    validation level (registered < runs < canonical < real_data)."""
    return _dump({"summary": catalog.load()["summary"],
                  "results": catalog.search(query, carrier, category, min_level, limit)})


@server.tool()
def imaging_modality(modality: str) -> str:
    """One modality's full design record: the eight spec.md fields, its forward-model templates and whether they
    run, its linked principle, validation level and gaps."""
    try:
        return _dump(catalog.get(modality))
    except KeyError as e:
        return _dump({"error": str(e)})


@server.tool()
def imaging_primitives(canonical: Optional[str] = None) -> str:
    """The registered primitives (building blocks of any design), each with its canonical primitive (P M Pi F C Sigma
    D S W R Lambda), physics stage, linearity and role. Filter by canonical letter."""
    rows = [core.primitive_info(k) for k in sorted(core.registry())]
    if canonical:
        rows = [r for r in rows if r["canonical"] == canonical]
    return _dump({"aliases": core.PRIMITIVE_ALIASES, "primitives": rows})


@server.tool()
def imaging_design_check(spec: Dict[str, Any], reference_modality: Optional[str] = None) -> str:
    """Check a new imaging design (spec.md: modality, carrier, geometry, object, forward_model{graph}, noise,
    target, system_elements). Runs C1 DAG, C2 canonical chain/novelty, C3 bounds, C4 nonlinear families, C5 adjoint,
    C6 representation error (against reference_modality's template if given), Gate 1 recoverability, Gate 2 carrier
    budget, Gate 3 calibration sensitivity and blind-calibration limits, and reports the eight adequacy dimensions
    only where the spec supplies them."""
    ref = None
    if reference_modality:
        tids = catalog.get(reference_modality)["spec_md"]["forward_model"]["graph_templates"]
        if tids:
            ref = core.contrib("graph_templates.yaml")["templates"][tids[0]]
    return _dump(design.check_design(spec, reference=ref))


@server.tool()
def imaging_design_submit(spec: Dict[str, Any], submitter: str) -> str:
    """File a design that passes the structural compiler as a pending digital-twin (L2) submission to the library.
    It is not added now: a named validator other than the submitter must accept it."""
    from ..library import Library
    rep = design.check_design(spec)
    if not rep.get("passed_structural"):
        return _dump({"error": "the design does not pass the structural compiler", "report": rep})
    lib = Library(Path(os.environ.get("PWM_MODE_HOME", Path.home() / ".ai4science-pwm-mode")) / "library")
    sub = design.submit_design(spec, rep, submitter, lib)
    return _dump({"submission": sub["id"], "state": "pending", "report": rep})


def _load(path: str):
    import numpy as np
    return np.load(Path(path))


def _save(arr, path: str) -> str:
    import numpy as np
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)
    return str(p)


@server.tool()
def imaging_lifecycle_design(modality: str) -> str:
    """Q4 for a registered modality: run the design check on its own spec.md record and forward model."""
    return _dump(lifecycle.design_check(modality))


@server.tool()
def imaging_simulate(modality: str, x_path: str, out_path: str, overrides: Optional[Dict[str, Any]] = None,
                     gain: float = 1.0) -> str:
    """Q3: y = forward model(x) for a registered modality (x: .npy in the template's object shape). Writes y (with the
    template's noise) to out_path and the clean linear output to out_path with '_clean' appended."""
    r = lifecycle.simulate(modality, _load(x_path), overrides, gain)
    clean = out_path.replace(".npy", "") + "_clean.npy"
    return _dump({"y": _save(r["y"], out_path), "y_clean": _save(r["y_clean"], clean), "template": r["template"]})


@server.tool()
def imaging_reconstruct(modality: str, y_path: str, out_path: str, overrides: Optional[Dict[str, Any]] = None,
                        gain: float = 1.0, lam: float = 1e-3, iters: int = 60) -> str:
    """Q1: x from y with the modality's operator (regularised least squares through a verified adjoint; the adjoint
    used, native or repaired, is reported). Writes x to out_path."""
    r = lifecycle.reconstruct(modality, _load(y_path), overrides, gain, lam, iters)
    if r["x"] is None:
        return _dump({k: v for k, v in r.items() if k != "x"})
    return _dump(dict({k: v for k, v in r.items() if k != "x"}, x=_save(r["x"], out_path)))


@server.tool()
def imaging_calibrate(modality: str, y_path: str, x_ref_path: Optional[str] = None, k: int = 9) -> str:
    """Q2: estimate the operator's calibratable parameters (and gain) from y. Blind without x_ref (gain refused: it
    is absorbable by the scene); reference mode with a known object x_ref (e.g. a calibration target)."""
    xr = _load(x_ref_path) if x_ref_path else None
    return _dump(lifecycle.calibrate(modality, _load(y_path), xr, k))


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
