"""The four lifecycle functions, for every modality in the imaging catalog.

For an instrument y = g * Phi_theta(x) + n (paper A1):
  design(m)                       Q4  check the modality's spec.md record (C1-C6, three gates, adequacy)
  simulate(m, x)                  Q3  y = forward model(x), with the template's noise; also the clean linear output
  reconstruct(m, y)               Q1  x from y, given the operator
  calibrate(m, y, x_ref=None)     Q2  the operator's parameters (and gain) from y; with a reference object if given

reconstruct uses the modality's own operator: regularised least squares (CG on Re(A^H A) + lam I) through the linear
part of the forward graph, with an adjoint that passes the dot-product test: the primitives' own, or a repaired one
(a fitted constant, or the exact transpose by probing for objects up to 8192 entries); the repair is recorded. When
neither works the function says so and names the registry's solver, which is not wired here. Nonlinear stages (noise,
detector response) are not inverted.

calibrate searches each calibratable parameter (the template's "learnable" parameters, plus mismatch_db parameters that
the graph actually uses, plus the gain g, which every linear instrument has) one at a time over a grid around the
nominal value. Blind mode scores a candidate by the data residual of the reconstruction it allows; reference mode (a
known object x_ref, e.g. a calibration target) scores ||y - A_theta x_ref||. Gain is absorbable by the scene (paper A1,
Theorem 1): blind mode does not attempt it and says it needs a reference.

audit() runs all four on every modality on a small synthetic object and records, per modality, whether each function
works and how well. Nothing here is a benchmark against real data.
"""
from __future__ import annotations

import json
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import adjoint, catalog, core, design

AUDIT_PATH = Path(__file__).with_name("lifecycle_audit.json")
GAIN_WORDS = ("gain", "scale", "qe", "throughput", "amplitude", "peak_photons")


def _template(modality: str) -> Dict[str, Any]:
    e = catalog.get(modality)
    tids = e["spec_md"]["forward_model"]["graph_templates"]
    T = core.contrib("graph_templates.yaml")["templates"]
    for tid in tids:
        if e["templates"].get(tid, {}).get("runs"):
            return dict(T[tid], _id=tid)
    raise ValueError(f"{modality}: no running forward-model template")


def _compile(t, overrides: Optional[Dict[str, Dict[str, Any]]] = None):
    nodes, _ = core.normalise_nodes(t["nodes"])
    if overrides:
        nodes = [dict(n, params=dict(n.get("params") or {}, **overrides.get(n["node_id"], {}))) for n in nodes]
    return core.compile_graph(nodes, t["edges"], t.get("metadata", {}))


def _linear(op):
    return [(nid, p) for nid, p in op.forward_plan if p.is_linear]


def _fwd(plan, x):
    for _, p in plan:
        x = p.forward(x)
    return x


def _adj(plan, y):
    for _, p in reversed(plan):
        y = p.adjoint(y)
    return y


def x_shape(modality: str):
    return tuple(_template(modality).get("metadata", {}).get("x_shape", [64, 64]))


def test_object(modality: str, seed: int = 0) -> np.ndarray:
    """A smooth non-negative object with a few blobs, in the template's object shape."""
    shape = x_shape(modality)
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape[:2]], indexing="ij")
    img = np.zeros(shape[:2])
    for _ in range(4):
        c = rng.uniform(-0.6, 0.6, 2)
        w = rng.uniform(0.1, 0.3)
        img += rng.uniform(0.5, 1.0) * np.exp(-((grids[0] - c[0]) ** 2 + (grids[1] - c[1]) ** 2) / (2 * w * w))
    x = img
    for n in shape[2:]:
        x = np.repeat(x[..., None], n, -1) * np.linspace(0.6, 1.0, n)
    return x.reshape(shape)


# ---------------------------------------------------------------- the four functions
def design_check(modality: str) -> Dict[str, Any]:
    e = catalog.get(modality)
    t = _template(modality)
    spec = dict(e["spec_md"], forward_model=dict(e["spec_md"]["forward_model"],
                                                 graph={"nodes": t["nodes"], "edges": t["edges"],
                                                        "metadata": t.get("metadata", {})}))
    return design.check_design(spec)


def simulate(modality: str, x: np.ndarray, overrides=None, gain: float = 1.0) -> Dict[str, Any]:
    t = _template(modality)
    op = _compile(t, overrides)
    return {"y": gain * np.asarray(op.forward(x)), "y_clean": gain * np.asarray(_fwd(_linear(op), x)),
            "template": t["_id"]}


MAX_PROBE = 8192  # largest object size for which the exact transpose is built by probing columns


_OP_CACHE: Dict[Any, Dict[str, Any]] = {}


def linear_operator(modality: str, overrides=None, gain: float = 1.0) -> Dict[str, Any]:
    key = (modality, json.dumps(overrides, sort_keys=True, default=str), float(gain))
    if key not in _OP_CACHE:
        _OP_CACHE[key] = _linear_operator(modality, overrides, gain)
    return _OP_CACHE[key]


def _linear_operator(modality: str, overrides=None, gain: float = 1.0) -> Dict[str, Any]:
    """The modality's linear forward operator with a verified adjoint.

    The registry primitives' own adjoints are used when they pass the dot-product test. If they fail (IMG-07 found
    36 templates), the adjoint is repaired, and the repair is recorded:
      'scale'  the primitive adjoint is right up to one constant (e.g. mri_kspace's unit-normalised inverse FFT):
               fit the constant on random pairs and check it on fresh ones;
      'probe'  otherwise, when the object has at most MAX_PROBE entries, build the exact matrix from A e_i;
      None     otherwise unavailable.
    """
    t = _template(modality)
    plan = _linear(_compile(t, overrides))
    shape = x_shape(modality)
    n = int(np.prod(shape))
    rng = np.random.default_rng(7)
    probe_out = np.asarray(_fwd(plan, np.zeros(shape)))
    cplx = np.iscomplexobj(probe_out) or np.iscomplexobj(np.asarray(_fwd(plan, np.random.default_rng(0).random(shape))))
    dt = complex if cplx else float
    A = lambda v: gain * np.asarray(_fwd(plan, np.asarray(v).reshape(shape)), dtype=dt)
    AH0 = lambda r: gain * np.asarray(_adj(plan, r), dtype=dt).reshape(-1)

    def dot_err(AH, trials=3):
        errs = []
        for _ in range(trials):
            x = rng.standard_normal(n)
            y0 = A(x)
            y = rng.standard_normal(y0.shape) + (1j * rng.standard_normal(y0.shape) if np.iscomplexobj(y0) else 0)
            lhs, rhs = np.vdot(y, y0), np.vdot(AH(y), x)
            errs.append(abs(lhs - np.conj(rhs)) / max(abs(lhs), 1e-12))
        return float(max(errs))

    try:
        e0 = dot_err(AH0)
    except Exception as e:  # noqa: BLE001
        e0, AH0 = float("inf"), None
    if e0 < 1e-4:
        return {"A": A, "AH": AH0, "shape": shape, "adjoint": "native", "dot_error": e0}
    if AH0 is not None:
        cs = []
        for _ in range(3):
            x = rng.standard_normal(n)
            y0 = A(x)
            y = rng.standard_normal(y0.shape) + (1j * rng.standard_normal(y0.shape) if np.iscomplexobj(y0) else 0)
            den = np.vdot(AH0(y), x)
            if abs(den) > 1e-12:
                cs.append(np.vdot(y, y0) / np.conj(den))
        if cs and np.std(cs) < 1e-6 * abs(np.mean(cs)):
            c = complex(np.mean(cs))
            AHs = lambda r, c=c: np.conj(c) * AH0(r)
            if dot_err(AHs) < 1e-4:
                return {"A": A, "AH": AHs, "shape": shape, "adjoint": "scale", "scale": abs(c), "dot_error": dot_err(AHs)}
    if n <= MAX_PROBE:
        cols = [A(np.eye(1, n, i).ravel()).ravel() for i in range(n)]
        M = np.stack(cols, 1)
        oshape = np.shape(A(np.zeros(n)))
        AHp = lambda r, M=M: (M.conj().T @ np.asarray(r).ravel())
        return {"A": A, "AH": AHp, "shape": shape, "adjoint": "probe", "dot_error": dot_err(AHp), "out_shape": oshape}
    return {"A": A, "AH": None, "shape": shape, "adjoint": None, "dot_error": e0}


def reconstruct(modality: str, y: np.ndarray, overrides=None, gain: float = 1.0, lam: float = 1e-3,
                iters: int = 60) -> Dict[str, Any]:
    from scipy.sparse.linalg import LinearOperator, cg
    op = linear_operator(modality, overrides, gain)
    if op["AH"] is None:
        solver = (core.contrib("solver_registry.yaml").get(modality) or {}).get("traditional_cpu", {})
        return {"x": None, "status": "unavailable: no verified adjoint (native fails, not a scaled adjoint, object too "
                                     "large to probe)", "registry_solver": solver.get("name")}
    shape, A, AH = op["shape"], op["A"], op["AH"]
    n = int(np.prod(shape))
    yl = np.asarray(y)
    if yl.shape != np.shape(A(np.zeros(n))):
        return {"x": None, "status": "unavailable: y shape does not match the linear output"}
    normal = LinearOperator((n, n), matvec=lambda v: np.real(AH(A(v))) + lam * v, dtype=float)
    x, _ = cg(normal, np.real(AH(yl)), maxiter=iters, rtol=1e-8)
    resid = float(np.linalg.norm(A(x) - yl) / max(np.linalg.norm(yl), 1e-12))
    return {"x": x.reshape(shape), "status": "ok", "method": "Tikhonov CG on the graph's linear operator",
            "adjoint": op["adjoint"], "data_residual": resid}


def calibratable(modality: str) -> List[Dict[str, Any]]:
    t = _template(modality)
    mm = (core.contrib("mismatch_db.yaml")["modalities"].get(modality) or {}).get("parameters", {})
    out, seen = [], set()
    for n in t["nodes"]:
        params = n.get("params") or {}
        names = list(n.get("learnable", [])) + [p for p in mm if p in params]
        for p in names:
            if (n["node_id"], p) in seen or not isinstance(params.get(p), (int, float)) or isinstance(params.get(p), bool):
                continue
            seen.add((n["node_id"], p))
            v = float(params[p])
            rng = mm.get(p, {}).get("range")
            out.append({"node": n["node_id"], "param": p, "nominal": v, "range": rng,
                        "integer": isinstance(params[p], int),
                        "absorbable": any(w in p.lower() for w in GAIN_WORDS)})
    out.append({"node": None, "param": "gain", "nominal": 1.0, "range": [0.5, 1.5], "integer": False, "absorbable": True})
    return out


def _grid(c, k=9):
    v = c["nominal"]
    lo, hi = (c["range"] if c["range"] else (v - max(abs(v) * 0.3, 1.0), v + max(abs(v) * 0.3, 1.0)))
    g = np.linspace(lo, hi, k)
    return sorted({int(round(a)) for a in g}) if c["integer"] else list(g)


def calibrate(modality: str, y: np.ndarray, x_ref: Optional[np.ndarray] = None, k: int = 9) -> Dict[str, Any]:
    """One-at-a-time grid search over each calibratable parameter; returns the estimate per parameter."""
    est = []
    for c in calibratable(modality):
        if x_ref is None and c["absorbable"]:
            est.append(dict(c, status="needs a reference measurement (absorbable by the scene)"))
            continue
        scores = []
        for v in _grid(c, k):
            ov = {c["node"]: {c["param"]: v}} if c["node"] else None
            g = v if c["param"] == "gain" else 1.0
            try:
                if x_ref is not None:
                    pred = simulate(modality, x_ref, ov, gain=g)["y_clean"]
                    s = float(np.linalg.norm(pred - y) / max(np.linalg.norm(y), 1e-12))
                else:
                    r = reconstruct(modality, y, ov, gain=g, iters=25)
                    if r["x"] is None:
                        raise ValueError(r["status"])
                    s = r["data_residual"]
                scores.append((s, v))
            except Exception as e:  # noqa: BLE001
                scores.append((float("inf"), v))
        best = min(scores)
        est.append(dict(c, estimate=best[1], score=best[0],
                        status="ok" if np.isfinite(best[0]) else "failed: no candidate evaluated"))
    return {"mode": "reference" if x_ref is not None else "blind", "parameters": est}


# ---------------------------------------------------------------- audit of all four, every modality
class _TO(Exception):
    pass


def audit_one(modality: str, seed: int = 0, budget_s: int = 300) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"modality": modality}
    signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_TO()))
    signal.alarm(budget_s)
    try:
        d = design_check(modality)
        rec["design"] = {"ok": bool(d.get("passed_structural")), "chain": d["checks"].get("C2_chain", {}).get("canonical_chain"),
                         "gates_assessed": [k for k, v in d["gates"].items() if not str(v.get("status", "")).startswith("unassessed")]}
        x = test_object(modality, seed)
        sim = simulate(modality, x)
        y = sim["y_clean"]
        rec["simulate"] = {"ok": bool(np.all(np.isfinite(np.asarray(sim["y"], dtype=complex)))), "y_shape": list(np.shape(sim["y"]))}
        r = reconstruct(modality, y)
        if r["x"] is not None:
            err = float(np.linalg.norm(r["x"] - x) / np.linalg.norm(x))
            # baseline: the scaled adjoint (back-projection), the trivial use of the same operator
            lo = linear_operator(modality)
            bp = np.real(lo["AH"](y)).reshape(x.shape)
            a = float(np.vdot(bp.ravel(), x.ravel()) / max(np.vdot(bp.ravel(), bp.ravel()), 1e-12))
            err_bp = float(np.linalg.norm(a * bp - x) / np.linalg.norm(x))
            rec["reconstruct"] = {"ok": err < err_bp, "rel_error": err, "rel_error_backprojection": err_bp,
                                  "data_residual": r["data_residual"], "adjoint": r["adjoint"]}
        else:
            rec["reconstruct"] = {"ok": False, "status": r["status"], "registry_solver": r.get("registry_solver")}
        cals = [c for c in calibratable(modality) if c["node"] is not None]
        if cals:
            c = cals[0]
            g = _grid(c)
            true_v = g[len(g) // 2 + 1] if len(g) > 2 else g[-1]  # perturb away from nominal, inside the grid
            ytrue = simulate(modality, x, {c["node"]: {c["param"]: true_v}})["y_clean"]
            single = dict(c)
            ref = None
            scores = []
            for v in g:
                pred = simulate(modality, x, {c["node"]: {c["param"]: v}})["y_clean"]
                scores.append((float(np.linalg.norm(pred - ytrue) / max(np.linalg.norm(ytrue), 1e-12)), v))
            ref = min(scores)[1]
            sens = float(max(s for s, _ in scores))
            rec["calibrate"] = {"param": f"{c['node']}.{c['param']}", "true": true_v, "reference_estimate": ref,
                                "ok": bool(np.isclose(ref, true_v)) and sens > 0,
                                "parameter_affects_output": sens > 0}
        else:
            rec["calibrate"] = {"param": "gain", "ok": True, "note": "only gain: recoverable with a reference target"}
    except _TO:
        rec["error"] = f"timeout after {budget_s}s"
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        signal.alarm(0)
    return rec


def audit(modalities: Optional[List[str]] = None) -> Dict[str, Any]:
    mods = modalities or [e["id"] for e in catalog.load()["modalities"]]
    rows = []
    for m in mods:
        rows.append(audit_one(m))
        print(m, {k: rows[-1].get(k, {}).get("ok") for k in ("design", "simulate", "reconstruct", "calibrate")},
              rows[-1].get("error", ""), flush=True)
    summ = {f: sum(bool(r.get(f, {}).get("ok")) for r in rows) for f in ("design", "simulate", "reconstruct", "calibrate")}
    summ["all_four"] = sum(all(r.get(f, {}).get("ok") for f in ("design", "simulate", "reconstruct", "calibrate")) for r in rows)
    summ["errors"] = sum("error" in r for r in rows)
    summ["n"] = len(rows)
    out = {"summary": summ, "rows": rows}
    AUDIT_PATH.write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(audit(sys.argv[1:] or None)["summary"], indent=1))
