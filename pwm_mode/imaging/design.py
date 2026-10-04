"""Design a new imaging modality and check it before it enters the library.

A design is a spec.md record with the eight fields of the system-design paper:
    modality, carrier, geometry, object, forward_model, noise, target, system_elements
where forward_model.graph = {"nodes": [...], "edges": [...], "metadata": {"x_shape": [...]}} is a typed DAG over the
registered primitives (101, mapping onto the 11 canonical primitives P M Pi F C Sigma D S W R Lambda), and
system_elements = {"source": {...}, "optics": [...], "detector": {...}, "calibration": [...], "cost": ...}.

check_design runs, in order:
  structural compiler   C1 DAG + registered primitives, C2 canonical chain (and whether it is new), C3 bounds
                        (N <= 20 nodes, depth <= 10), C4 nonlinear nodes restricted to Detect/Transform/Scatter
                        families, C5 adjoint dot-product test on the linear part, C6 representation error against a
                        reference graph when one is given
  Gate 1 (recoverability)   compression ratio m/n, the null-space dimension it forces, the exact numerical rank
                            when min(m, n) <= 512, and a pass/fail only against target.min_compression_ratio
  Gate 2 (carrier budget)   SNR = QE N / sqrt(QE N + read^2 + dark t) from system_elements (source photons,
                            quantum efficiency, read noise, dark current, exposure)
  Gate 3 (operator mismatch) forward-output sensitivity to each declared calibration parameter at its tolerance,
                            and the calibration modes that no blind calibration can recover (absorbable symmetries:
                            detector gain always; translation in single-pixel-like M->Sigma and convolutional C chains)
  adequacy (flagship, eight dimensions) reported only where the design supplies the inputs; never invented
A check that lacks its inputs is reported as "unassessed", not passed. submit_design files a passing design as a
pending L2 (digital twin) submission; it enters the library only through Library.accept with an independent validator.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from . import core

SPEC_FIELDS = ("modality", "carrier", "geometry", "object", "forward_model", "noise", "target", "system_elements")
CARRIERS = ("photon", "xray", "electron", "acoustic", "spin", "particle", "thz", "neutron")
ADEQUACY = ("acquisition_feasibility", "temporal_adequacy", "spatial_adequacy", "observable_sufficiency",
            "output_recovery_quality", "budget_feasibility", "deployment_burden", "sample_compatibility")
NONLINEAR_CANONICAL = {"D", "Lambda", "R"}


def _depth(nodes, edges) -> int:
    ids = [n["node_id"] for n in nodes]
    succ = {i: [] for i in ids}
    indeg = {i: 0 for i in ids}
    for e in edges:
        succ[e["source"]].append(e["target"])
        indeg[e["target"]] += 1
    depth = {i: 1 for i in ids}
    queue = [i for i in ids if indeg[i] == 0]
    while queue:
        u = queue.pop()
        for v in succ[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return max(depth.values()) if depth else 0


def _linear_part(op):
    """The forward plan with stochastic and non-linear terminal noise stripped (noise is Gate 2, not the operator)."""
    return [(nid, p) for nid, p in op.forward_plan if p.is_linear]


def _apply(plan, x, adjoint=False):
    cur = x
    for nid, p in (reversed(plan) if adjoint else plan):
        cur = p.adjoint(cur) if adjoint else p.forward(cur)
    return cur


def check_design(spec: Dict[str, Any], reference: Optional[Dict[str, Any]] = None, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    rep: Dict[str, Any] = {"modality": spec.get("modality"), "checks": {}, "gates": {}, "adequacy": {}, "notes": []}
    missing = [f for f in SPEC_FIELDS if f not in spec]
    rep["checks"]["spec_fields"] = {"pass": not missing, "missing": missing}
    if spec.get("carrier") not in CARRIERS:
        rep["notes"].append(f"carrier {spec.get('carrier')!r} is not one of {CARRIERS}")
    g = (spec.get("forward_model") or {}).get("graph")
    if not g:
        rep["checks"]["C1_dag"] = {"pass": False, "error": "forward_model.graph (nodes, edges) is required"}
        rep["passed_structural"] = False
        return rep
    nodes, aliases = core.normalise_nodes(g["nodes"])
    if aliases:
        rep["notes"].append("primitive aliases applied: " + "; ".join(aliases))
    meta = dict(g.get("metadata") or {})
    try:
        op = core.compile_graph(nodes, g["edges"], meta)
        rep["checks"]["C1_dag"] = {"pass": True}
    except Exception as e:  # noqa: BLE001
        rep["checks"]["C1_dag"] = {"pass": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        rep["passed_structural"] = False
        return rep
    chain = op.canonical_dag_string()
    known = core.canonical_decompositions()
    same = sorted(k for k, v in known.items() if v == chain)
    implicit_d = sorted(k for k, v in known.items() if v == chain + " -> D")  # templates often leave Detect implicit
    rep["checks"]["C2_chain"] = {"pass": True, "canonical_chain": chain, "matches_validated": same + implicit_d,
                                 "terminal_detect_implicit": bool(implicit_d and not same),
                                 "novel_composition": not (same or implicit_d)}
    n, depth = len(op.forward_plan), _depth(nodes, g["edges"])
    rep["checks"]["C3_bounds"] = {"pass": n <= core.N_MAX and depth <= core.D_MAX, "nodes": n, "depth": depth}
    bad = []
    for nid, p in op.forward_plan:
        info = core.primitive_info(p.primitive_id)
        if not info["linear"] and info["canonical"] is not None and info["canonical"] not in NONLINEAR_CANONICAL:
            bad.append(f"{nid} ({p.primitive_id}, {info['canonical']})")
    rep["checks"]["C4_nonlinear"] = {"pass": not bad, "violations": bad}
    xs = tuple(meta.get("x_shape", [64, 64]))
    lin = _linear_part(op)
    try:
        x = rng.random(xs)
        y = _apply(lin, x)
        v = rng.random(np.shape(y))
        lhs = np.vdot(_apply(lin, x), v)
        rhs = np.vdot(x, _apply(lin, v, adjoint=True))
        rel = abs(lhs - rhs) / max(abs(lhs), 1e-8)
        rep["checks"]["C5_adjoint"] = {"pass": bool(rel < 1e-4), "relative_error": float(rel),
                                       "linear_nodes": [nid for nid, _ in lin]}
    except Exception as e:  # noqa: BLE001
        rep["checks"]["C5_adjoint"] = {"pass": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    if reference:
        try:
            rn, _ = core.normalise_nodes(reference["nodes"])
            ref = core.compile_graph(rn, reference["edges"], reference.get("metadata") or meta)
            errs = []
            for _ in range(5):
                x = rng.random(xs)
                a, b = _apply(lin, x), _apply(_linear_part(ref), x)
                errs.append(float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)))
            rep["checks"]["C6_representation"] = {"pass": float(np.mean(errs)) < 0.01, "epsilon": float(np.mean(errs))}
        except Exception as e:  # noqa: BLE001
            rep["checks"]["C6_representation"] = {"pass": False, "error": str(e)[:200]}
    else:
        rep["checks"]["C6_representation"] = {"pass": None, "status": "unassessed: no reference forward model"}
    try:
        y0 = op.forward(rng.random(xs))
        rep["checks"]["forward_runs"] = {"pass": bool(np.all(np.isfinite(np.asarray(y0, dtype=complex)))),
                                         "y_shape": list(np.shape(y0))}
    except Exception as e:  # noqa: BLE001
        rep["checks"]["forward_runs"] = {"pass": False, "error": str(e)[:200]}
    rep["passed_structural"] = all(c.get("pass") is not False for c in rep["checks"].values())

    # ---- Gate 1: recoverability
    try:
        nx = int(np.prod(xs))
        my = int(np.prod(np.shape(_apply(lin, rng.random(xs)))))
        gamma = my / nx
        g1 = {"compression_ratio_m_over_n": gamma, "null_space_dim_at_least": max(nx - my, 0)}
        k = min(nx, my)
        if k <= 512:  # exact rank on a full probe set is affordable; otherwise only the counting bound is reported
            probes = np.stack([np.asarray(_apply(lin, rng.standard_normal(xs)), dtype=complex).ravel()
                               for _ in range(k)], 1)
            sv = np.linalg.svd(probes, compute_uv=False)
            g1["numerical_rank"] = int(np.sum(sv > sv[0] * 1e-8))
            g1["rank_deficient"] = g1["numerical_rank"] < k
        gmin = (spec.get("target") or {}).get("min_compression_ratio")
        g1["pass"] = (gamma >= gmin) if gmin is not None else None
        g1["status"] = ("assessed against target.min_compression_ratio" if gmin is not None else
                        "underdetermined: recovery relies on the object prior; give target.min_compression_ratio "
                        "(Compression Bound threshold) to assess" if gamma < 1 else "not underdetermined")
        rep["gates"]["G1_recoverability"] = g1
    except Exception as e:  # noqa: BLE001
        rep["gates"]["G1_recoverability"] = {"status": "unassessed", "error": str(e)[:200]}

    # ---- Gate 2: carrier budget
    se = spec.get("system_elements") or {}
    src, det = se.get("source") or {}, se.get("detector") or {}
    if isinstance(src, list):
        src = src[0] if src else {}
    if isinstance(det, list):
        det = det[0] if det else {}
    N = src.get("photons_per_pixel") or src.get("carrier_count_per_pixel")
    qe, read = det.get("quantum_efficiency"), det.get("read_noise_e")
    if N and qe is not None and read is not None:
        dark = (det.get("dark_current_e_per_s") or 0.0) * (det.get("exposure_s") or 0.0)
        snr = qe * N / math.sqrt(qe * N + read ** 2 + dark)
        need = (spec.get("target") or {}).get("min_snr")
        rep["gates"]["G2_carrier_budget"] = {"snr": snr, "snr_db": 20 * math.log10(max(snr, 1e-12)),
                                             "pass": (snr >= need) if need else None,
                                             "status": "assessed" if need else "SNR computed; no min_snr target given"}
    else:
        rep["gates"]["G2_carrier_budget"] = {"status": "unassessed: system_elements needs source.photons_per_pixel, "
                                                       "detector.quantum_efficiency and detector.read_noise_e"}

    # ---- Gate 3: operator mismatch and calibration
    calib = se.get("calibration") or []
    sens = []
    for c in calib:
        node, param, tol = c.get("node"), c.get("parameter"), c.get("tolerance")
        if node is None or tol is None:
            sens.append({"parameter": param, "status": "unassessed: needs node and tolerance"})
            continue
        try:
            pert = [dict(n_, params=dict(n_.get("params") or {}, **{param: (n_.get("params") or {}).get(param, 0) + tol}))
                    if n_["node_id"] == node else n_ for n_ in nodes]
            op2 = core.compile_graph(pert, g["edges"], meta)
            x = rng.random(xs)
            a, b = _apply(lin, x), _apply(_linear_part(op2), x)
            change = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
            row = {"parameter": f"{node}.{param}", "tolerance": tol, "relative_output_change": change}
            if change == 0.0:
                row["status"] = "no effect: the primitive may not use this parameter name; check its params"
            sens.append(row)
        except Exception as e:  # noqa: BLE001
            sens.append({"parameter": f"{node}.{param}", "error": str(e)[:160]})
    canon = chain.split(" -> ")
    blind = ["detector gain (absorbable by scene scaling): needs a reference-target record"]
    if canon[:2] == ["M", "Sigma"] or canon[:1] == ["C"]:
        blind.append("translation (absorbable by scene translation in this chain): needs a per-session reference")
    rep["gates"]["G3_mismatch"] = {"sensitivity": sens, "not_blindly_calibratable": blind,
                                   "status": "assessed" if sens else "unassessed: declare system_elements.calibration"}

    # ---- eight adequacy dimensions (flagship): only what the design supplies
    task = spec.get("task_requirements") or {}
    for dim in ADEQUACY:
        rep["adequacy"][dim] = task.get(dim, "unassessed")
    if not se.get("source"):
        rep["notes"].append("no source element: source and excitation design cannot be assessed")
    return rep


def submit_design(spec: Dict[str, Any], report: Dict[str, Any], submitter: str, library, field: str = "imaging"):
    """File a structurally passing design as a pending L2 (digital-twin) submission. It joins the library only when
    a named validator other than the submitter accepts it (Library.accept)."""
    if not report.get("passed_structural"):
        raise ValueError("only a design that passes the structural compiler can be submitted")
    mid = str(spec["modality"]).lower().replace("_", "-")
    chain = report["checks"]["C2_chain"]["canonical_chain"]
    return library.submit(field=field, layer="L2", id=f"img-design-{mid}", title=f"Design: {spec['modality']}",
                          summary=f"{spec.get('carrier')} | {chain} | novel={report['checks']['C2_chain']['novel_composition']}",
                          payload={"spec_md": spec, "check_report": report},
                          conditions={"carrier": spec.get("carrier"), "canonical_chain": chain},
                          source_problems=[], submitter=submitter,
                          evidence={"structural": True, "gates": {k: v.get("status") for k, v in report["gates"].items()}})
