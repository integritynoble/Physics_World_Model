"""Dead-parameter finder for pwm_core graph templates (LIB-40 IMG-10 part 2).

A template parameter is dead when changing it leaves the forward output of the compiled graph unchanged, because the
primitive never reads that name or reads it and does not use it. Two checks:

* `scan_templates`: for every template and every node, perturb each numeric/bool entry of `params` and each name in
  `learnable`, compile, run `forward` on a fixed random input, and compare with the unperturbed output.
* `static_unused_reads`: for every registered primitive, find `v = self._params.get("name", ...)` whose `v` is never used.

Small inputs only (the templates' own 64 x 64), one process.
Run: PWM_CORE_SRC=<pwm_core> nice -n 19 python3 -m pwm_mode.imaging.sensitivity --out results/IMG-10/sensitivity.json
"""
from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import signal
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import numpy as np

from . import core

REL_TOL = 1e-12
SHAPE_KEYS = {"x_shape", "y_shape", "shape", "H", "W", "mode", "seed"}
MISSING_TRY = (1.0, 2.0)


class _Timeout(Exception):
    pass


_ADJOINT_ERROR = "adjoint-error"


def _perturb(v: Any) -> List[Any]:
    """Several perturbations, so that a parameter that is saturated, clipped or below a pixel at its nominal value is not called dead."""
    if isinstance(v, bool):
        return [not v]
    if isinstance(v, int):
        return [v + 1, max(v - 1, 0)] if v < 4 else [int(round(v * 1.5)), max(v // 2, 1), max(v // 10, 1)]
    if isinstance(v, float):
        return [1.0, 0.1] if v == 0 else [v * 1.37, v * 0.7, v * 5.0, v * 0.1, v * 20.0]
    if isinstance(v, (list, tuple)) and v and all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in v):
        return [[e * f if e else f for e in v] for f in (1.37, 0.7)]
    return []


def _forward(nodes, edges, meta, x, yref=None):
    """Forward output; with `yref`, also the adjoint of `yref` through the linear nodes only (noise and other nonlinear nodes are skipped; None if that fails)."""
    nodes, _ = core.normalise_nodes(nodes)
    signal.alarm(20)
    try:
        op = core.compile_graph(nodes, edges, meta)
        y = np.asarray(op.forward(x))
        if yref is None:
            return y
        try:
            a = yref
            for _, prim in op.adjoint_plan:
                if prim.is_linear:
                    a = prim.adjoint(a)
            a = np.asarray(a)
        except Exception:  # noqa: BLE001
            a = _ADJOINT_ERROR
        return y, a
    except _Timeout as e:
        raise RuntimeError("timeout") from e
    finally:
        signal.alarm(0)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    d = np.nanmax(np.abs(a - b)) if a.size else 0.0
    return float(d / (np.nanmax(np.abs(a)) + 1e-300))


def _verdict(base, other) -> Dict[str, Any]:
    ry = _rel(base[0], other[0])
    if isinstance(base[1], np.ndarray) and other[1] is _ADJOINT_ERROR:
        ra = float("inf")  # the adjoint no longer runs: the parameter is tied to the operator's shapes
    elif isinstance(base[1], np.ndarray) and isinstance(other[1], np.ndarray):
        ra = _rel(base[1], other[1])
    else:
        ra = 0.0
    r = max(ry, ra)
    where = [n for n, v in (("forward", ry), ("adjoint", ra)) if v >= REL_TOL]
    return {"status": "dead" if r < REL_TOL else "acts", "rel_change": r, "acts_in": where}


def _case(template: Dict[str, Any], x: np.ndarray, base, ni: int, name: str, values: List[Any]) -> Dict[str, Any]:
    results = []
    for q in values:
        nodes = copy.deepcopy(template["nodes"])
        nodes[ni]["params"][name] = q
        try:
            other = _forward(nodes, template["edges"], template.get("metadata", {}), x, base[0])
        except Exception as e:  # noqa: BLE001 - recorded
            results.append({"status": "error", "error": f"{type(e).__name__}: {str(e)[:120]}"})
            continue
        results.append(_verdict(base, other))
    ok = [r for r in results if r["status"] != "error"]
    if not ok:
        return results[0]
    return max(ok, key=lambda r: r["rel_change"])


def _retry_dead(template: Dict[str, Any], x: np.ndarray, row: Dict[str, Any], ni: int, name: str, values: List[Any]):
    """A parameter that did not act may be inactive only because of the template's input or a sibling parameter.

    * `needs_axis`: it acts once the object has a trailing axis of length 4 (spectral / time / depth), which this
      template's 2-D `x_shape` does not have.
    * `gated`: it acts once the other zero-valued numeric parameters of its node are set to 0.5.
    """
    meta = template.get("metadata", {})
    x3 = np.random.default_rng(2).random(x.shape + (4,))
    try:
        y0 = _forward(copy.deepcopy(template["nodes"]), template["edges"], meta, x3)
        base3 = _forward(copy.deepcopy(template["nodes"]), template["edges"], meta, x3, y0)
        res = _case(template, x3, base3, ni, name, values)
        if res["status"] == "acts":
            return "needs_axis", res
    except Exception:  # noqa: BLE001
        pass
    node = template["nodes"][ni]
    zeros = [k for k, v in (node.get("params") or {}).items()
             if k != name and isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0]
    if zeros:
        t2 = copy.deepcopy(template)
        for k in zeros:
            t2["nodes"][ni]["params"][k] = 0.5
        try:
            y0 = _forward(copy.deepcopy(t2["nodes"]), t2["edges"], meta, x)
            base2 = _forward(copy.deepcopy(t2["nodes"]), t2["edges"], meta, x, y0)
            res = _case(t2, x, base2, ni, name, values)
            if res["status"] == "acts":
                return "gated", dict(res, gated_by=zeros)
        except Exception:  # noqa: BLE001
            pass
    return None, None


def scan_template(tid: str, template: Dict[str, Any]) -> Dict[str, Any]:
    meta = template.get("metadata", {})
    x = np.random.default_rng(0).random(tuple(meta.get("x_shape", [64, 64])))
    try:
        y0 = _forward(copy.deepcopy(template["nodes"]), template["edges"], meta, x)
        base = _forward(copy.deepcopy(template["nodes"]), template["edges"], meta, x, y0)
        y0b = base[0]
    except Exception as e:  # noqa: BLE001
        return {"template": tid, "modality": meta.get("modality"), "runs": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}
    if _rel(y0, y0b) > 0:
        return {"template": tid, "modality": meta.get("modality"), "runs": True, "deterministic": False}
    try:
        x2 = np.random.default_rng(1).random(x.shape)
        ignores_input = _rel(y0, _forward(copy.deepcopy(template["nodes"]), template["edges"], meta, x2)) < REL_TOL
    except Exception:  # noqa: BLE001
        ignores_input = None
    mag = float(np.nanmax(np.abs(y0))) if y0.size else 0.0
    spread = float(np.nanstd(y0)) if y0.size else 0.0
    output = {"shape": list(y0.shape), "max_abs": mag, "std": spread, "degenerate": bool(spread <= 1e-12 * max(mag, 1e-300)),
              "ignores_object": ignores_input}
    rows: List[Dict[str, Any]] = []
    for ni, node in enumerate(template["nodes"]):
        params = node.get("params") or {}
        learnable = list(node.get("learnable") or [])
        for name, v in params.items():
            if name in ("seed",) or name in SHAPE_KEYS and not isinstance(v, (int, float)):
                continue
            pv = _perturb(v)
            if not pv:
                continue
            res = _case(template, x, base, ni, name, pv)
            if res["status"] == "dead":
                kind, res2 = _retry_dead(template, x, res, ni, name, pv)
                if kind:
                    res = dict(res2, status=kind)
            rows.append({"node": node["node_id"], "primitive": node["primitive_id"], "param": name,
                         "learnable": name in learnable, "in_params": True, **res})
        for name in learnable:
            if name in params:
                continue
            outs = []
            for val in MISSING_TRY:
                try:
                    nodes = copy.deepcopy(template["nodes"])
                    nodes[ni]["params"][name] = val
                    outs.append(_forward(nodes, template["edges"], meta, x, y0))
                except Exception as e:  # noqa: BLE001
                    outs.append(e)
            if any(isinstance(o, Exception) for o in outs):
                res = {"status": "error", "error": str([o for o in outs if isinstance(o, Exception)][0])[:120]}
            else:
                res = _verdict(outs[0], outs[1])
            rows.append({"node": node["node_id"], "primitive": node["primitive_id"], "param": name,
                         "learnable": True, "in_params": False, **res})
    return {"template": tid, "modality": meta.get("modality"), "runs": True, "deterministic": True, "output": output, "params": rows}


def scan_templates(only: Optional[List[str]] = None) -> Dict[str, Any]:
    signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Timeout()))
    templates = core.contrib("graph_templates.yaml")["templates"]
    out = []
    for tid, t in templates.items():
        if only and tid not in only:
            continue
        out.append(scan_template(tid, t))
    return {"templates": out}


def static_unused_reads() -> List[Dict[str, Any]]:
    found = []
    for pid, cls in sorted(core.registry().items()):
        for mname in ("forward", "adjoint", "__init__"):
            fn = cls.__dict__.get(mname)
            if fn is None:
                continue
            try:
                tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            except (OSError, TypeError, SyntaxError):
                continue
            body = tree.body[0]
            loads = {n.id for n in ast.walk(body) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            for node in ast.walk(body):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "get" and node.value.args
                        and isinstance(node.value.args[0], ast.Constant) and isinstance(node.value.args[0].value, str)
                        and "_params" in ast.dump(node.value.func.value)):
                    var = node.targets[0].id
                    uses = sum(1 for n in ast.walk(body) if isinstance(n, ast.Name) and n.id == var
                               and isinstance(n.ctx, ast.Load))
                    if uses == 0:
                        found.append({"primitive": pid, "method": mname, "param": node.value.args[0].value, "variable": var})
    return found


def summarise(report: Dict[str, Any]) -> Dict[str, Any]:
    rows = [dict(r, template=t["template"]) for t in report["templates"] for r in t.get("params", [])]
    dead = [r for r in rows if r["status"] == "dead"]
    by = {}
    for r in dead:
        by.setdefault((r["primitive"], r["param"], r["learnable"]), []).append(r["template"])
    cond = {}
    for r in rows:
        if r["status"] in ("needs_axis", "gated"):
            cond.setdefault((r["status"], r["primitive"], r["param"]), []).append(r["template"])
    return {"templates": len(report["templates"]), "templates_run": sum(1 for t in report["templates"] if t["runs"]),
            "nondeterministic": [t["template"] for t in report["templates"] if t.get("deterministic") is False],
            "failed": [{"template": t["template"], "error": t["error"]} for t in report["templates"] if not t["runs"]],
            "params_tested": len(rows), "dead": len(dead),
            "dead_learnable": sum(1 for r in dead if r["learnable"]),
            "conditional": [{"status": k[0], "primitive": k[1], "param": k[2], "templates": v}
                            for k, v in sorted(cond.items())],
            "errors": sum(1 for r in rows if r["status"] == "error"),
            "dead_by_primitive_param": [{"primitive": k[0], "param": k[1], "learnable": k[2], "count": len(v),
                                         "templates": v} for k, v in sorted(by.items())]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    t0 = time.time()
    rep = scan_templates(a.only)
    rep["static_unused_reads"] = static_unused_reads()
    rep["summary"] = summarise(rep)
    rep["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    json.dump(rep["summary"], sys.stdout, indent=1)


if __name__ == "__main__":
    main()
