"""Run validation experiments across all registry modalities.

Generates agent-like forward model specs for each modality in the canonical
decomposition registry, translates them, compiles them through the
6-gate compiler, and reports results.

Usage:
    PYTHONPATH=packages/pwm_core python3 papers/system_design/compiler/run_validation.py
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, "packages/pwm_core")

from pwm_core.graph.canonical_decompositions import CANONICAL_DECOMPOSITIONS, CanonicalDecomposition
from pwm_core.graph.ir_types import CanonicalPrimitive

from papers.system_design.compiler.agent_translator import AgentToGraphTranslator
from papers.system_design.compiler.primitive_compiler import (
    CompilationReport,
    ConstrainedPrimitiveCompiler,
)


# ---------------------------------------------------------------------------
# Synthetic agent spec generators (one per modality pattern)
# ---------------------------------------------------------------------------

def _make_source(carrier: str) -> Dict[str, Any]:
    """Generate a source element based on carrier type."""
    name_map = {
        "photon": ("photon_src", "Photon Source"),
        "xray": ("xray_src", "X-ray Source"),
        "electron": ("electron_src", "Electron Beam"),
        "acoustic": ("acoustic_src", "Acoustic Source"),
        "spin": ("spin_src", "Spin Excitation"),
        "particle": ("particle_src", "Particle Source"),
    }
    eid, name = name_map.get(carrier, ("generic_src", "Source"))
    return {"id": eid, "type": "source", "name": name, "parameters": {}}


def _prim_to_element(prim: CanonicalPrimitive, idx: int) -> Dict[str, Any]:
    """Convert a canonical primitive to a FlowchartElement."""
    # Each entry: (name_suffix, element_type, parameters)
    # Must produce hints that the AgentToGraphTranslator will correctly route
    mapping = {
        CanonicalPrimitive.P: ("propagation", "geometry", {"model": "fresnel"}),
        CanonicalPrimitive.M: ("modulation", "interaction", {"model": "absorption"}),
        CanonicalPrimitive.Pi: ("projection", "geometry", {"model": "radon", "n_angles": 180}),
        CanonicalPrimitive.F: ("kspace_encoding", "geometry", {"model": "kspace"}),
        CanonicalPrimitive.C: ("convolution", "geometry", {"model": "psf"}),
        CanonicalPrimitive.Sigma: ("spectral_accumulation", "detector", {"model": "intensity"}),
        CanonicalPrimitive.D: ("detector", "detector", {"model": "intensity"}),
        CanonicalPrimitive.S: ("sampling", "geometry", {"model": "undersampling"}),
        CanonicalPrimitive.W: ("dispersion", "geometry", {"model": "dispersion"}),
        CanonicalPrimitive.R: ("scatter", "interaction", {"model": "scatter"}),
        CanonicalPrimitive.Lambda: ("transform", "interaction", {"model": "beer_lambert"}),
    }
    name_suffix, el_type, params = mapping.get(prim, ("unknown", "other", {}))
    return {
        "id": f"{name_suffix}_{idx}",
        "type": el_type,
        "name": f"{prim.name} ({name_suffix})",
        "parameters": params,
    }


def generate_agent_spec(modality: str, decomp: CanonicalDecomposition) -> Dict[str, Any]:
    """Generate a synthetic agent spec matching the canonical decomposition."""
    elements = []

    # Source element
    source = _make_source(decomp.carrier)
    elements.append(source)

    # Canonical primitives as elements
    for i, prim in enumerate(decomp.primitives):
        el = _prim_to_element(prim, i)
        elements.append(el)

    # Wire connects_to sequentially
    for i in range(len(elements) - 1):
        elements[i]["connects_to"] = [elements[i + 1]["id"]]
    elements[-1]["connects_to"] = []

    return {
        "elements": elements,
        "measurement_shape": "(64, 64)",
    }


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def run_validation() -> None:
    translator = AgentToGraphTranslator()
    compiler = ConstrainedPrimitiveCompiler()

    results: List[Dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    n_chain_match = 0

    print(f"{'Modality':<25} {'Chain':<30} {'Valid':>6} {'Match':>6} {'Nodes':>6} {'Depth':>6} {'Time':>8}")
    print("-" * 95)

    for modality, decomp in sorted(CANONICAL_DECOMPOSITIONS.items()):
        # Generate synthetic agent spec
        action = generate_agent_spec(modality, decomp)

        # Translate
        try:
            spec = translator.translate(action, modality=modality)
        except Exception as e:
            print(f"{modality:<25} {'TRANSLATE_ERROR':<30} {'FAIL':>6} {'N/A':>6} {'--':>6} {'--':>6} {'--':>8}")
            results.append({"modality": modality, "valid": False, "error": f"translate: {e}"})
            n_fail += 1
            continue

        # Compile
        report = compiler.compile(spec, modality=modality)

        chain_match = "Yes" if report.canonical_match else "No"
        valid_str = "PASS" if report.valid else "FAIL"

        if report.valid:
            n_pass += 1
        else:
            n_fail += 1
        if report.canonical_match:
            n_chain_match += 1

        print(
            f"{modality:<25} {report.canonical_chain_str:<30} {valid_str:>6} "
            f"{chain_match:>6} {report.node_count:>6} {report.depth:>6} "
            f"{report.compilation_time_s * 1000:>7.2f}ms"
        )

        result = {
            "modality": modality,
            "valid": report.valid,
            "canonical_chain": report.canonical_chain_str,
            "canonical_match": report.canonical_match,
            "node_count": report.node_count,
            "depth": report.depth,
            "nonlinear_ok": report.nonlinear_ok,
            "failures": report.failures,
            "warnings": report.warnings,
            "compilation_time_ms": round(report.compilation_time_s * 1000, 2),
            "expected_chain": decomp.dag,
            "carrier": decomp.carrier,
            "validation_level": decomp.validation_level,
        }
        results.append(result)

    # Summary
    total = n_pass + n_fail
    print("-" * 95)
    print(f"Total: {total} | Pass: {n_pass} ({100*n_pass/total:.1f}%) | "
          f"Fail: {n_fail} ({100*n_fail/total:.1f}%) | "
          f"Chain match: {n_chain_match} ({100*n_chain_match/total:.1f}%)")

    # Save results
    output_path = "papers/system_design/compiler/validation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate markdown table for paper
    md_path = "papers/system_design/compiler/validation_results.md"
    with open(md_path, "w") as f:
        f.write("# Constrained Primitive Compiler Validation Results\n\n")
        f.write(f"**Total modalities**: {total}\n")
        f.write(f"**Pass rate**: {100*n_pass/total:.1f}%\n")
        f.write(f"**Chain fidelity**: {100*n_chain_match/total:.1f}%\n\n")
        f.write("| Modality | Expected Chain | Compiled Chain | Valid | Match | Carrier | Level |\n")
        f.write("|----------|---------------|----------------|-------|-------|---------|-------|\n")
        for r in results:
            valid = "PASS" if r.get("valid") else "FAIL"
            match = "Yes" if r.get("canonical_match") else "No"
            f.write(
                f"| {r['modality']} | {r.get('expected_chain', 'N/A')} | "
                f"{r.get('canonical_chain', 'N/A')} | {valid} | {match} | "
                f"{r.get('carrier', '?')} | {r.get('validation_level', '?')} |\n"
            )
    print(f"Markdown table saved to {md_path}")


if __name__ == "__main__":
    run_validation()
