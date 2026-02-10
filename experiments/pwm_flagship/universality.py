"""PWM Flagship -- Universality: compile all 26 graph templates.

For each template in ``contrib/graph_templates.yaml``:

1. Compile to OperatorGraph (inject graph_id from template name,
   strip 'description' which is forbidden by StrictBaseModel).
2. Validate schema (OperatorGraphSpec).
3. Serialize to JSON.
4. Run adjoint check (where applicable -- only for fully linear graphs).

Output: pass/fail table (modality, compiled?, schema_valid?, serializable?,
adjoint_pass?).

Usage::

    PYTHONPATH=. python -m experiments.pwm_flagship.universality --out_dir results/flagship_universality
    PYTHONPATH=. python -m experiments.pwm_flagship.universality --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from pwm_core.graph.compiler import GraphCompiler, GraphCompilationError
from pwm_core.graph.graph_spec import OperatorGraphSpec

logger = logging.getLogger(__name__)

TEMPLATES_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "packages" / "pwm_core" / "contrib" / "graph_templates.yaml"
)

# All 26 expected modalities
EXPECTED_MODALITIES = [
    "widefield", "widefield_lowdose", "confocal_livecell",
    "confocal_3d", "sim", "lightsheet", "cassi", "spc", "cacti",
    "matrix", "ct", "mri", "ptychography", "holography",
    "nerf", "gaussian_splatting", "lensless", "panorama",
    "light_field", "dot", "photoacoustic", "oct", "flim",
    "fpm", "phase_retrieval", "integral",
]


def _load_templates() -> Dict[str, Any]:
    """Load all templates from the YAML registry."""
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("templates", {})


def compile_one_template(
    template_id: str,
    template: Dict[str, Any],
    compiler: GraphCompiler,
) -> Dict[str, Any]:
    """Compile, validate, serialize, and adjoint-check one template.

    Returns a result dict with pass/fail flags.
    """
    result: Dict[str, Any] = {
        "template_id": template_id,
        "compiled": False,
        "schema_valid": False,
        "serializable": False,
        "adjoint_pass": None,  # None = not applicable (non-linear)
        "adjoint_rel_err": None,
        "all_linear": None,
        "n_nodes": 0,
        "n_edges": 0,
        "error": None,
    }

    try:
        # Strip 'description' (forbidden by StrictBaseModel with extra="forbid")
        template_clean = {
            k: v for k, v in template.items() if k != "description"
        }

        # Inject graph_id from template name
        template_clean["graph_id"] = template_id

        # Step 1: Validate schema
        spec = OperatorGraphSpec.model_validate(template_clean)
        result["schema_valid"] = True

        # Step 2: Compile
        graph_op = compiler.compile(spec)
        result["compiled"] = True
        result["n_nodes"] = len(graph_op.forward_plan)
        result["n_edges"] = len(graph_op.edges)
        result["all_linear"] = graph_op.all_linear

        # Step 3: Serialize
        serialized = graph_op.serialize()
        json_str = json.dumps(serialized, default=str)
        assert len(json_str) > 0
        result["serializable"] = True

        # Step 4: Adjoint check (only for fully linear graphs)
        if graph_op.all_linear:
            report = graph_op.check_adjoint(n_trials=3, rtol=1e-3, seed=42)
            result["adjoint_pass"] = report.passed
            result["adjoint_rel_err"] = report.max_relative_error
        else:
            result["adjoint_pass"] = None  # Not applicable

    except Exception as exc:
        result["error"] = str(exc)
        logger.warning("Template %s failed: %s", template_id, exc)

    return result


def run_universality(
    out_dir: str,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Compile all 26 templates and output pass/fail table."""
    os.makedirs(out_dir, exist_ok=True)
    compiler = GraphCompiler()
    templates = _load_templates()

    if smoke:
        # Only first 5 templates in smoke mode
        template_ids = sorted(templates.keys())[:5]
    else:
        template_ids = sorted(templates.keys())

    results: List[Dict[str, Any]] = []

    for tid in template_ids:
        logger.info("Universality: compiling %s", tid)
        r = compile_one_template(tid, templates[tid], compiler)
        results.append(r)

    # Summary statistics
    n_total = len(results)
    n_compiled = sum(1 for r in results if r["compiled"])
    n_schema = sum(1 for r in results if r["schema_valid"])
    n_serial = sum(1 for r in results if r["serializable"])
    n_adjoint_applicable = sum(1 for r in results if r["adjoint_pass"] is not None)
    n_adjoint_pass = sum(1 for r in results if r["adjoint_pass"] is True)

    summary = {
        "total_templates": n_total,
        "compiled": n_compiled,
        "schema_valid": n_schema,
        "serializable": n_serial,
        "adjoint_applicable": n_adjoint_applicable,
        "adjoint_pass": n_adjoint_pass,
        "all_pass": n_compiled == n_total and n_schema == n_total and n_serial == n_total,
        "results": results,
    }

    # Print table
    logger.info("=" * 80)
    logger.info("Universality Results: %d/%d compiled, %d/%d schema, "
                "%d/%d serial, %d/%d adjoint",
                n_compiled, n_total, n_schema, n_total,
                n_serial, n_total, n_adjoint_pass, n_adjoint_applicable)
    logger.info("=" * 80)
    logger.info("%-35s %8s %8s %8s %8s", "Template", "Compiled", "Schema", "Serial", "Adjoint")
    logger.info("-" * 80)
    for r in results:
        adj_str = (
            "PASS" if r["adjoint_pass"] is True
            else "FAIL" if r["adjoint_pass"] is False
            else "N/A"
        )
        logger.info(
            "%-35s %8s %8s %8s %8s",
            r["template_id"],
            "PASS" if r["compiled"] else "FAIL",
            "PASS" if r["schema_valid"] else "FAIL",
            "PASS" if r["serializable"] else "FAIL",
            adj_str,
        )

    # Save
    with open(os.path.join(out_dir, "universality_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Universality: %d/%d templates pass all checks -> %s",
                n_compiled, n_total, out_dir)
    return results


# -- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PWM Flagship: Universality -- compile all 26 templates"
    )
    parser.add_argument("--out_dir", default="results/flagship_universality")
    parser.add_argument("--smoke", action="store_true",
                        help="Only first 5 templates")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_universality(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
