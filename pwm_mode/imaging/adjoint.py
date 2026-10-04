"""C5 adjoint (dot-product) test for compiled graph templates.

For a linear operator A the adjoint A^H must satisfy <Ax, y> = <x, A^H y> for all x, y. The test draws seeded
random x and y, evaluates both sides and reports the relative error |lhs - rhs| / max(|lhs|, |rhs|).

What is tested, per template: the forward plan is split into maximal runs of consecutive primitives that report
``is_linear``. Each run is a linear operator on its own input (shape and dtype taken from a real forward pass) and is
tested separately, so a non-linear primitive in the middle of a chain (Beer-Lambert, |.|^2, ...) is never silently
dropped or tied across. A template whose plan is one run is ``linear``; one whose only non-linear primitives come last
(noise, detection) is ``terminal_nonlinear`` and its single run is the whole noise-free operator; otherwise it is
``interior_nonlinear`` and several runs are tested.

Inner products: real x gives the real inner product on Re(<Ax, y>); complex x gives the complex one. When it fails,
the diagnostics separate the likely causes (not linear in fact, no adjoint, shape mismatch, non-deterministic
forward, adjoint that drops the imaginary part).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import core

# Relative error |lhs-rhs|/max(|lhs|,|rhs|). Float64 chains that are exact adjoints give <= 3e-14 here; every failing one
# gives >= 6e-2, so any value from 1e-12 to 1e-2 yields the same verdicts. 1e-8 leaves six orders of margin each way
# (pwm_core's own check_adjoint and design.py C5 use 1e-4).
TOLERANCE = 1e-8
N_TRIALS = 3
DEFAULT_SEED = 20261005


def classify(flags: List[bool]) -> str:
    if all(flags):
        return "linear"
    first = flags.index(False)
    return "terminal_nonlinear" if not any(flags[first:]) else "interior_nonlinear"


def runs(flags: List[bool]) -> List[Tuple[int, int]]:
    """Half-open index ranges of maximal consecutive linear primitives."""
    out, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    return out


def _rand(rng, like: np.ndarray) -> np.ndarray:
    a = rng.standard_normal(np.shape(like))
    if np.iscomplexobj(like):
        a = a + 1j * rng.standard_normal(np.shape(like))
    return a


def _fwd(plan, x):
    for _, p in plan:
        x = p.forward(x)
    return x


def _adj(plan, y):
    for _, p in reversed(plan):
        y = p.adjoint(y)
    return y


def _norm(a) -> float:
    return float(np.linalg.norm(np.asarray(a).ravel()))


def dot_test(plan, x0: np.ndarray, seed: int, n_trials: int = N_TRIALS) -> Dict[str, Any]:
    """Dot-product test of the linear chain ``plan`` at input shape/dtype of ``x0``."""
    rng = np.random.default_rng(seed)
    errs: List[float] = []
    ratios: List[float] = []
    out: Dict[str, Any] = {"n_trials": n_trials}
    try:
        for _ in range(n_trials):
            x = _rand(rng, x0)
            ax = np.asarray(_fwd(plan, x))
            y = _rand(rng, ax)
            aty = np.asarray(_adj(plan, y))
            if aty.shape != x.shape:
                out.update(status="fail", reason="adjoint output shape differs from input shape",
                           detail=f"x {x.shape}, adjoint(y) {aty.shape}")
                if aty.size == x.size:
                    r = aty.reshape(x.shape)
                    lh, rh = (np.vdot(ax, y), np.vdot(x, r)) if np.iscomplexobj(x) else (np.vdot(ax, y).real, np.vdot(x, r).real)
                    out["error_if_reshaped"] = float(abs(lh - rh) / max(abs(lh), abs(rh), 1e-30))
                return out
            if np.iscomplexobj(x):
                lhs, rhs = np.vdot(ax, y), np.vdot(x, aty)
            else:
                lhs, rhs = np.vdot(ax, y).real, np.vdot(x, aty).real
            denom = max(abs(lhs), abs(rhs), 1e-30)
            errs.append(float(abs(lhs - rhs) / denom))
            if abs(lhs) > 1e-30:
                ratios.append(float(np.real(rhs / lhs)))
            if np.iscomplexobj(x):
                lr, rr = np.vdot(ax, y).real, np.vdot(x, aty).real
                out["real_inner_product_error"] = max(out.get("real_inner_product_error", 0.0),
                                                      float(abs(lr - rr) / max(abs(lr), abs(rr), 1e-30)))
            if np.iscomplexobj(ax) and not np.iscomplexobj(x) and np.iscomplexobj(y):
                full = abs(np.vdot(ax, y) - np.vdot(x, aty)) / max(abs(np.vdot(ax, y)), 1e-30)
                out["imag_part_error"] = max(out.get("imag_part_error", 0.0), float(full))
    except NotImplementedError as e:
        out.update(status="fail", reason="adjoint not implemented", detail=str(e)[:200])
        return out
    except Exception as e:  # noqa: BLE001 - the error text is the record
        out.update(status="fail", reason="exception", detail=f"{type(e).__name__}: {str(e)[:200]}")
        return out
    out["max_relative_error"] = max(errs)
    if ratios and max(errs) >= TOLERANCE:
        m = float(np.mean(ratios))
        out["rhs_over_lhs"] = {"mean": m, "constant": bool(abs(m) > 0 and np.std(ratios) / abs(m) < 1e-6)}
    out["status"] = "pass" if max(errs) < TOLERANCE and np.isfinite(max(errs)) else "fail"
    if out["status"] == "fail":
        out["reason"] = "dot-product mismatch"
    return out


def linearity(plan, x0: np.ndarray, seed: int) -> Dict[str, Any]:
    """Is the chain linear in fact: A(a x1 + b x2) = a A x1 + b A x2, A(0) = 0, and forward deterministic."""
    rng = np.random.default_rng(seed + 1)
    try:
        x1, x2 = _rand(rng, x0), _rand(rng, x0)
        a, b = 1.7, -0.6
        lhs = np.asarray(_fwd(plan, a * x1 + b * x2))
        rhs = a * np.asarray(_fwd(plan, x1)) + b * np.asarray(_fwd(plan, x2))
        scale = max(_norm(lhs), _norm(rhs), 1e-30)
        zero = _norm(_fwd(plan, np.zeros_like(x1)))
        again = _norm(np.asarray(_fwd(plan, x1)) - np.asarray(_fwd(plan, x1)))
        return {"additivity_error": _norm(lhs - rhs) / scale, "response_to_zero": zero / scale,
                "repeat_difference": again / max(_norm(_fwd(plan, x1)), 1e-30)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:150]}"}


def node_diagnosis(plan, x0: np.ndarray, seed: int) -> List[Dict[str, Any]]:
    """Dot-test each primitive of a failing chain on its own (input taken from the chain)."""
    out, cur = [], x0
    for i, (nid, p) in enumerate(plan):
        r = dot_test([(nid, p)], cur, seed + 100 + i)
        out.append({"node": nid, "primitive": p.primitive_id, "status": r["status"],
                    "max_relative_error": r.get("max_relative_error"), "reason": r.get("reason"),
                    "detail": r.get("detail"), "real_inner_product_error": r.get("real_inner_product_error"),
                    "error_if_reshaped": r.get("error_if_reshaped")})
        try:
            cur = np.asarray(p.forward(cur))
        except Exception:  # noqa: BLE001
            break
    return out


def test_template(template: Dict[str, Any], seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Compile one graph template and dot-test every maximal linear run. Never raises."""
    rec: Dict[str, Any] = {"modality": (template.get("metadata") or {}).get("modality")}
    try:
        nodes, notes = core.normalise_nodes(template["nodes"])
        op = core.compile_graph(nodes, template["edges"], template.get("metadata", {}))
    except Exception as e:  # noqa: BLE001
        rec.update(status="compile_error", error=f"{type(e).__name__}: {str(e)[:200]}")
        return rec
    plan = list(op.forward_plan)
    flags = [bool(p.is_linear) for _, p in plan]
    rec.update(aliases_applied=notes, x_shape=list(op.x_shape), primitives=[p.primitive_id for _, p in plan],
               linear_flags=flags, kind=classify(flags))
    rng = np.random.default_rng(seed)
    cur = rng.random(tuple(op.x_shape))
    inputs = []
    try:
        for _, p in plan:
            inputs.append(cur)
            cur = np.asarray(p.forward(cur))
    except Exception as e:  # noqa: BLE001
        rec.update(status="forward_error", error=f"{type(e).__name__}: {str(e)[:200]}")
        return rec
    segs = []
    for k, (a, b) in enumerate(runs(flags)):
        sub = plan[a:b]
        r = dot_test(sub, inputs[a], seed + 10 * k)
        r.update(nodes=[n for n, _ in sub], primitives=[p.primitive_id for _, p in sub], input_shape=list(inputs[a].shape),
                 input_complex=bool(np.iscomplexobj(inputs[a])))
        if r["status"] != "pass":
            r["linearity"] = linearity(sub, inputs[a], seed + 10 * k)
            r["per_node"] = node_diagnosis(sub, inputs[a], seed + 10 * k)
        segs.append(r)
    rec["segments"] = segs
    rec["status"] = "pass" if segs and all(s["status"] == "pass" for s in segs) else ("no_linear_part" if not segs else "fail")
    return rec


def load_templates() -> Dict[str, Dict[str, Any]]:
    return core.contrib("graph_templates.yaml")["templates"]


def design_c5(template: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
    """What the design checker's C5 says about this template (design.check_design's own helpers, same seed rule)."""
    from . import design
    try:
        nodes, _ = core.normalise_nodes(template["nodes"])
        op = core.compile_graph(nodes, template["edges"], template.get("metadata", {}))
        lin = design._linear_part(op)
        rng = np.random.default_rng(seed)
        x = rng.random(tuple((template.get("metadata") or {}).get("x_shape", [64, 64])))
        v = rng.random(np.shape(design._apply(lin, x)))
        lhs, rhs = np.vdot(design._apply(lin, x), v), np.vdot(x, design._apply(lin, v, adjoint=True))
        rel = abs(lhs - rhs) / max(abs(lhs), 1e-8)
        return {"pass": bool(rel < 1e-4), "relative_error": float(rel), "linear_nodes": [n for n, _ in lin]}
    except Exception as e:  # noqa: BLE001
        return {"pass": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def run_all(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    out = {}
    for name, t in load_templates().items():
        r = test_template(t, seed)
        r["design_c5"] = design_c5(t)
        out[name] = r
    return out


if __name__ == "__main__":  # python -m pwm_mode.imaging.adjoint --out results.json
    import argparse
    import json
    import warnings
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    res = run_all(a.seed)
    json.dump({"tolerance": TOLERANCE, "n_trials": N_TRIALS, "seed": a.seed, "templates": res}, open(a.out, "w"),
              indent=1, default=str)
    from collections import Counter
    print(dict(Counter(r["status"] for r in res.values())))
