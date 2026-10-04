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

from . import catalog, core, design

server = MCPServer(
    name="ai4science-pwm-imaging",
    version="0.1.0",
    instructions=(
        "PWM imaging library. imaging_search finds registered modalities (170, each a spec.md design record with its "
        "validation level and gaps); imaging_modality returns one record; imaging_primitives lists the building "
        "blocks; imaging_design_check compiles a new design and runs the structural compiler (C1-C6) and the three "
        "gates; imaging_design_submit files a passing design for independent validation. Report a check's own "
        "status: 'unassessed' is not a pass."),
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


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
