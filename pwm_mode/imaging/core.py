"""Access to the PWM imaging physics (pwm_core) from PWM mode.

The forward-model compiler, the 101 registered primitives and the 170-modality registry live in the full pwm_core
package (Physics_World_Model/packages/pwm_core); the copy vendored inside AI4Science has no graph compiler. Set
PWM_CORE_SRC to point elsewhere.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

PWM_CORE_SRC = Path(os.environ.get("PWM_CORE_SRC", "/home/spiritai/pwm/Physics_World_Model/packages/pwm_core"))
GENESIS_SRC = Path(os.environ.get("PWM_GENESIS_SRC", "/home/spiritai/pwm/Physics_World_Model/pwm_product/genesis"))

# Primitive ids used by some registry templates that were never registered; each has a registered equivalent.
# (Found by the 5 Oct 2026 coverage audit: 20 modalities' templates failed to compile only because of these names.)
PRIMITIVE_ALIASES = {
    "propagation_2d": "angular_spectrum",
    "radon_2d": "ct_radon",
    "fft_2d": "fourier_relay",
    "poisson_gaussian_sensor_noise": "poisson_gaussian_sensor",
}

# Bounds from the Finite Primitive Basis theorem.
N_MAX, D_MAX = 20, 10


def _ensure_path() -> None:
    p = str(PWM_CORE_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


@lru_cache(maxsize=1)
def registry():
    _ensure_path()
    from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
    return PRIMITIVE_REGISTRY


def contrib(name: str):
    import yaml
    return yaml.safe_load(open(PWM_CORE_SRC / "contrib" / name))


def primitive_info(pid: str) -> Dict[str, Any]:
    cls = registry()[pid]
    cid = getattr(cls, "_canonical_id", None)
    stage = getattr(cls, "_physics_stage", None)
    return {"id": pid, "canonical": cid.name if cid is not None else None,
            "stage": stage.value if stage is not None else None,
            "role": getattr(cls, "_node_role", None), "linear": bool(getattr(cls, "_is_linear", False)),
            "stochastic": bool(getattr(cls, "_is_stochastic", False)),
            "detect_family": str(getattr(cls, "_detect_family", None) or "") or None,
            "tier": getattr(cls, "_physics_tier", None),
            "summary": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""}


def normalise_nodes(nodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Apply PRIMITIVE_ALIASES; return the nodes and the list of substitutions made."""
    out, notes = [], []
    for n in nodes:
        pid = n.get("primitive_id")
        if pid in PRIMITIVE_ALIASES:
            notes.append(f"{n.get('node_id')}: {pid} -> {PRIMITIVE_ALIASES[pid]}")
            n = dict(n, primitive_id=PRIMITIVE_ALIASES[pid])
        out.append(n)
    return out, notes


def compile_graph(nodes, edges, metadata=None):
    _ensure_path()
    from pwm_core.graph.compiler import GraphCompiler
    from pwm_core.graph.graph_spec import OperatorGraphSpec
    spec = OperatorGraphSpec(graph_id=(metadata or {}).get("graph_id", "design"), nodes=nodes, edges=edges,
                             metadata=metadata or {})
    return GraphCompiler().compile(spec)


def canonical_decompositions() -> Dict[str, str]:
    _ensure_path()
    from pwm_core.graph.canonical_decompositions import CANONICAL_DECOMPOSITIONS
    return {k: " -> ".join(p.name for p in v.primitives) for k, v in CANONICAL_DECOMPOSITIONS.items()}
