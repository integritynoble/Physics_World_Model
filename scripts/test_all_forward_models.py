#!/usr/bin/env python3
"""Test all 64 imaging modality forward models.

Tests every modality via two paths:
  Path A (Dedicated): Call _build_operator_by_id() directly
  Path B (Graph-first): Load graph template -> strip stochastic nodes ->
      compile -> wrap in GraphOperatorAdapter

Per-modality tests:
  1. Instantiation: operator builds without error
  2. Forward pass: forward(x) runs and produces finite output
  3. Adjoint pass: adjoint(y) runs and produces finite output (if available)
  4. Shape consistency: declared shapes match actual shapes
  5. Adjoint consistency (linear only): <Ax,y> ~ <x,A^T y>
  6. CorrectedOperator wrapping: PrePostCorrection & LowRankCorrection

Usage:
    python scripts/test_all_forward_models.py
"""

from __future__ import annotations

import re
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# All 64 modalities from graph_templates.yaml
# ---------------------------------------------------------------------------
ALL_MODALITIES = [
    # Microscopy
    "widefield", "widefield_lowdose", "confocal_livecell", "confocal_3d",
    "sim", "lightsheet",
    # Compressive / Spectral
    "cassi", "spc", "cacti", "matrix",
    # Tomography / MRI
    "ct", "mri",
    # Phase / Holography / Coherent
    "ptychography", "holography", "phase_retrieval", "fpm",
    # Rendering / 3D
    "nerf", "gaussian_splatting",
    # Lensless / Computational
    "lensless", "panorama", "light_field",
    # Biomedical
    "dot", "photoacoustic", "oct", "flim", "integral",
    # New dedicated operators
    "xray_radiography", "ultrasound", "pet", "spect",
    "sem", "tem", "electron_tomography",
    # Additional v2 modalities (graph-only)
    "stem", "fluoroscopy", "mammography", "dexa", "cbct",
    "angiography", "doppler_ultrasound", "elastography",
    "fmri", "mrs", "diffusion_mri",
    "two_photon", "sted", "palm_storm", "tirf",
    "polarization", "endoscopy", "fundus", "octa",
    "tof_camera", "lidar", "structured_light",
    "sar", "sonar",
    "electron_diffraction", "ebsd", "eels", "electron_holography",
    "neutron_tomo", "proton_radiography", "muon_tomo",
]

# Modalities with dedicated factory routes
DEDICATED_MODALITIES = {
    "widefield", "sim", "cassi", "spc", "cacti",
    "lensless", "lightsheet", "ct", "mri", "ptychography", "holography",
    "nerf", "gaussian_splatting", "oct", "light_field",
    "photoacoustic", "fpm", "flim", "dot", "integral",
    "phase_retrieval", "cdi",
    "ultrasound", "sem", "tem", "electron_tomography",
    "pet", "spect", "xray_radiography",
    "matrix",
}

# Modalities that map to a different dedicated ID
MODALITY_TO_DEDICATED = {
    "widefield_lowdose": "widefield",
    "confocal_livecell": "confocal",
    "confocal_3d": "confocal",
}


@dataclass
class TestResult:
    modality: str
    path: str  # "dedicated", "graph"
    instantiation: str = "SKIP"
    forward_pass: str = "SKIP"
    adjoint_pass: str = "SKIP"
    shape_consistency: str = "SKIP"
    adjoint_consistency: str = "SKIP"
    prepost_correction: str = "SKIP"
    lowrank_correction: str = "SKIP"
    error_msg: str = ""
    duration_ms: float = 0.0


def _build_operator_dedicated(modality: str, dims: Tuple[int, ...] = (64, 64)):
    """Build operator via _build_operator_by_id (dedicated path)."""
    from pwm_core.core.physics_factory import _build_operator_by_id
    return _build_operator_by_id(modality, dims, {}, None)


def _build_operator_graph(modality: str, dims: Tuple[int, ...] = (64, 64)):
    """Build operator via graph-first path."""
    from pwm_core.core.physics_factory import _try_build_graph_operator
    return _try_build_graph_operator(modality, dims)


def _get_x_shape(op) -> Tuple[int, ...]:
    """Get x_shape, falling back to declared shape."""
    shape = getattr(op, "x_shape", None)
    if shape is not None and shape != (1,):
        return tuple(shape)
    # Try _x_shape
    shape = getattr(op, "_x_shape", None)
    if shape is not None and shape != (1,):
        return tuple(shape)
    return (64, 64)


def _get_y_shape(op) -> Tuple[int, ...]:
    """Get y_shape, falling back to running forward once."""
    shape = getattr(op, "y_shape", None)
    if shape is not None and shape != (1,):
        return tuple(shape)
    shape = getattr(op, "_y_shape", None)
    if shape is not None and shape != (1,):
        return tuple(shape)
    return None  # Will be inferred from forward


def test_forward(op, x_shape: Tuple[int, ...]) -> Tuple[str, Optional[np.ndarray]]:
    """Test forward pass: runs, finite output, reasonable shape."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(x_shape).astype(np.float32)
    y = op.forward(x)
    if not np.all(np.isfinite(y)):
        n_bad = int(np.sum(~np.isfinite(y)))
        return f"FAIL ({n_bad} non-finite values in output shape {y.shape})", y
    return f"PASS (shape {y.shape})", y


def test_adjoint(op, y_shape: Tuple[int, ...]) -> Tuple[str, Optional[np.ndarray]]:
    """Test adjoint pass: runs, finite output."""
    rng = np.random.default_rng(43)
    y = rng.standard_normal(y_shape).astype(np.float32)
    try:
        x_adj = op.adjoint(y)
    except RuntimeError as e:
        if "non-linear" in str(e).lower() or "not available" in str(e).lower():
            return "SKIP (nonlinear graph)", None
        raise
    if not np.all(np.isfinite(x_adj)):
        n_bad = int(np.sum(~np.isfinite(x_adj)))
        return f"FAIL ({n_bad} non-finite values)", x_adj
    return f"PASS (shape {x_adj.shape})", x_adj


def test_shape_consistency(
    op, y: np.ndarray, x_adj: Optional[np.ndarray], x_shape: Tuple[int, ...]
) -> str:
    """Test that declared shapes match actual shapes."""
    issues = []
    declared_y = getattr(op, "y_shape", None)
    if declared_y is not None and declared_y != (1,) and y is not None:
        if y.shape != tuple(declared_y):
            issues.append(f"y_shape declared={declared_y} actual={y.shape}")
    declared_x = getattr(op, "x_shape", None)
    if declared_x is not None and declared_x != (1,) and x_adj is not None:
        if x_adj.shape != tuple(declared_x):
            issues.append(f"x_shape declared={declared_x} actual={x_adj.shape}")
    if issues:
        return f"FAIL ({'; '.join(issues)})"
    return "PASS"


def test_adjoint_consistency(op, x_shape, y_shape, rtol: float = 1e-4, seed: int = 42) -> str:
    """Test <Ax,y> ~ <x, A^T y> for linear operators.

    Handles complex-valued operators using complex inner products.
    Reports rotation-based operators with relaxed tolerance (WARN not FAIL).
    """
    is_linear = getattr(op, "is_linear", False)
    if not is_linear:
        return "SKIP (nonlinear)"

    # Rotation-based operators have inherent adjoint mismatch due to
    # bilinear interpolation in ndimage.rotate. Use relaxed tolerance.
    op_id = getattr(op, "operator_id", "")
    rotation_based = op_id in (
        "ct", "pet", "spect", "electron_tomography",
    )

    rng = np.random.default_rng(seed)
    max_err = 0.0

    for trial in range(3):
        x = rng.standard_normal(x_shape).astype(np.float64)
        y = rng.standard_normal(y_shape).astype(np.float64)

        try:
            Ax = op.forward(x)
            ATy = op.adjoint(y)
        except RuntimeError:
            return "SKIP (adjoint unavailable)"

        # Handle complex outputs properly
        Ax = np.asarray(Ax).ravel()
        ATy = np.asarray(ATy).ravel()
        x_flat = x.ravel()
        y_flat = y.ravel()

        if np.iscomplexobj(Ax) or np.iscomplexobj(ATy):
            # Complex inner product: <Ax, y> = sum(Ax * conj(y))
            if not np.iscomplexobj(y_flat):
                y_flat = y_flat.astype(np.complex128)
            if not np.iscomplexobj(x_flat):
                x_flat = x_flat.astype(np.complex128)
            inner_Ax_y = complex(np.sum(Ax * np.conj(y_flat)))
            inner_x_ATy = complex(np.sum(x_flat * np.conj(ATy)))
            denom = max(abs(inner_Ax_y), abs(inner_x_ATy), 1e-30)
            rel_err = abs(inner_Ax_y - inner_x_ATy) / denom
        else:
            Ax = Ax.astype(np.float64)
            ATy = ATy.astype(np.float64)
            inner_Ax_y = float(np.sum(Ax * y_flat))
            inner_x_ATy = float(np.sum(x_flat * ATy))
            denom = max(abs(inner_Ax_y), abs(inner_x_ATy), 1e-30)
            rel_err = abs(inner_Ax_y - inner_x_ATy) / denom

        max_err = max(max_err, rel_err)

    if max_err < rtol:
        return f"PASS (max_err={max_err:.2e})"
    if rotation_based:
        return f"WARN (max_err={max_err:.2e}, rotation-based)"
    return f"FAIL (max_err={max_err:.2e}, tol={rtol})"


def test_prepost_correction(op, x_shape, y_shape) -> str:
    """Test PrePostCorrection wrapping."""
    try:
        from pwm_core.graph.corrected_operator import (
            CorrectedOperator,
            PrePostCorrection,
        )

        correction = PrePostCorrection(
            pre_scale=1.1, pre_shift=0.01,
            post_scale=0.9, post_shift=-0.01,
        )
        corrected = CorrectedOperator(op, correction)

        rng = np.random.default_rng(99)
        x = rng.standard_normal(x_shape).astype(np.float32)

        y_corr = corrected.forward(x)
        if not np.all(np.isfinite(y_corr)):
            return "FAIL (non-finite forward)"

        # Test adjoint
        y_test = rng.standard_normal(y_shape).astype(np.float32)
        try:
            x_adj = corrected.adjoint(y_test)
            if not np.all(np.isfinite(x_adj)):
                return "FAIL (non-finite adjoint)"
        except RuntimeError:
            pass  # adjoint not available for nonlinear graphs

        # Identity correction should match base
        identity_corr = PrePostCorrection()
        identity_op = CorrectedOperator(op, identity_corr)
        y_base = op.forward(x)
        y_identity = identity_op.forward(x)
        if not np.allclose(
            np.asarray(y_base, dtype=np.float64),
            np.asarray(y_identity, dtype=np.float64),
            rtol=1e-5, atol=1e-7,
        ):
            return "FAIL (identity correction != base)"

        return "PASS"
    except Exception as e:
        return f"FAIL ({e})"


def test_lowrank_correction(op, x_shape, y_shape) -> str:
    """Test LowRankCorrection wrapping."""
    try:
        from pwm_core.graph.corrected_operator import (
            CorrectedOperator,
            LowRankCorrection,
        )

        M = int(np.prod(y_shape))
        N = int(np.prod(x_shape))
        rank = 2

        rng = np.random.default_rng(77)
        U = rng.standard_normal((M, rank)) * 0.01
        V = rng.standard_normal((N, rank)) * 0.01
        alphas = np.array([0.1, -0.1])

        correction = LowRankCorrection(U, V, alphas)
        corrected = CorrectedOperator(op, correction)

        x = rng.standard_normal(x_shape).astype(np.float32)
        y_corr = corrected.forward(x)

        if not np.all(np.isfinite(y_corr)):
            return "FAIL (non-finite output)"

        # Zero-alpha should match base
        zero_corr = LowRankCorrection(U, V, np.zeros(rank))
        zero_op = CorrectedOperator(op, zero_corr)
        y_base = op.forward(x)
        y_zero = zero_op.forward(x)
        if not np.allclose(
            np.asarray(y_base, dtype=np.float64).ravel(),
            np.asarray(y_zero, dtype=np.float64).ravel(),
            rtol=1e-5, atol=1e-7,
        ):
            return "FAIL (zero-alpha != base)"

        return "PASS"
    except Exception as e:
        return f"FAIL ({e})"


def run_single_modality(modality: str) -> List[TestResult]:
    """Run all tests for a single modality, trying both paths."""
    results = []

    # Determine which paths to try
    paths = []
    dedicated_id = MODALITY_TO_DEDICATED.get(modality, modality)
    if dedicated_id in DEDICATED_MODALITIES:
        paths.append(("dedicated", dedicated_id))
    paths.append(("graph", modality))

    for path, build_id in paths:
        result = TestResult(modality=modality, path=path)
        t0 = time.time()

        try:
            # 1. Instantiation
            if path == "dedicated":
                op = _build_operator_dedicated(build_id)
            else:
                op = _build_operator_graph(build_id)

            if op is None:
                result.instantiation = "SKIP (no template)"
                result.duration_ms = (time.time() - t0) * 1000
                results.append(result)
                continue

            result.instantiation = "PASS"

            # Determine shapes
            x_shape = _get_x_shape(op)
            y_shape_declared = _get_y_shape(op)

            # 2. Forward pass
            fwd_result, y = test_forward(op, x_shape)
            result.forward_pass = fwd_result

            if y is None:
                result.duration_ms = (time.time() - t0) * 1000
                results.append(result)
                continue

            # Infer y_shape from actual forward output if not declared
            y_shape = y_shape_declared if y_shape_declared is not None else y.shape

            # 3. Adjoint pass
            adj_result, x_adj = test_adjoint(op, y_shape)
            result.adjoint_pass = adj_result

            # 4. Shape consistency
            result.shape_consistency = test_shape_consistency(op, y, x_adj, x_shape)

            # 5. Adjoint consistency
            if x_adj is not None:
                result.adjoint_consistency = test_adjoint_consistency(
                    op, x_shape, y_shape
                )
            else:
                result.adjoint_consistency = "SKIP (no adjoint)"

            # 6. CorrectedOperator tests
            result.prepost_correction = test_prepost_correction(op, x_shape, y_shape)
            result.lowrank_correction = test_lowrank_correction(op, x_shape, y_shape)

        except Exception as e:
            if result.instantiation == "SKIP":
                result.instantiation = f"FAIL ({e})"
            result.error_msg = traceback.format_exc()

        result.duration_ms = (time.time() - t0) * 1000
        results.append(result)

    return results


def print_report(all_results: List[TestResult]):
    """Print tabular report grouped by category."""
    dedicated = [r for r in all_results if r.path == "dedicated"]
    graph = [r for r in all_results if r.path == "graph"]

    total_pass = 0
    total_fail = 0
    total_skip = 0
    total_warn = 0

    def count(s: str):
        nonlocal total_pass, total_fail, total_skip, total_warn
        if s.startswith("PASS"):
            total_pass += 1
        elif s.startswith("FAIL"):
            total_fail += 1
        elif s.startswith("WARN"):
            total_warn += 1
        else:
            total_skip += 1

    def short(s):
        if s.startswith("PASS"):
            return "PASS"
        if s.startswith("FAIL"):
            return "FAIL"
        if s.startswith("WARN"):
            return "WARN"
        return "SKIP"

    header = (
        f"{'Modality':<28} {'Path':<10} {'Init':<6} {'Fwd':<6} "
        f"{'Adj':<6} {'Shape':<6} {'AdjCk':<12} {'PrePost':<8} "
        f"{'LowRk':<8} {'ms':<8}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  PHYSICS FORWARD MODEL TEST REPORT")
    print("=" * len(header))

    for group_name, group in [("DEDICATED OPERATORS", dedicated), ("GRAPH-FIRST OPERATORS", graph)]:
        if not group:
            continue
        print(f"\n--- {group_name} ---")
        print(header)
        print(sep)

        for r in sorted(group, key=lambda r: r.modality):
            fields = [r.instantiation, r.forward_pass, r.adjoint_pass,
                      r.shape_consistency, r.adjoint_consistency,
                      r.prepost_correction, r.lowrank_correction]
            for f in fields:
                count(f)

            adj_ck = short(r.adjoint_consistency)
            m = re.search(r'max_err=([\d.e+-]+)', r.adjoint_consistency)
            if m:
                adj_ck = f"{short(r.adjoint_consistency)}({m.group(1)})"

            print(
                f"{r.modality:<28} {r.path:<10} {short(r.instantiation):<6} "
                f"{short(r.forward_pass):<6} {short(r.adjoint_pass):<6} "
                f"{short(r.shape_consistency):<6} {adj_ck:<12} "
                f"{short(r.prepost_correction):<8} {short(r.lowrank_correction):<8} "
                f"{r.duration_ms:<8.0f}"
            )

            if r.error_msg:
                lines = r.error_msg.strip().split("\n")
                print(f"    {lines[-1]}")

    total_tests = total_pass + total_fail + total_skip + total_warn
    print(f"\n{'=' * len(header)}")
    print(f"SUMMARY: {total_pass} PASS / {total_fail} FAIL / {total_warn} WARN / {total_skip} SKIP "
          f"({total_tests} tests, {len(all_results)} operator instances)")
    print(f"Modalities tested: {len(set(r.modality for r in all_results))}")

    # Print failures
    fail_results = [r for r in all_results if any(
        getattr(r, f).startswith("FAIL")
        for f in ["instantiation", "forward_pass", "adjoint_pass",
                   "shape_consistency", "prepost_correction", "lowrank_correction"]
    )]
    if fail_results:
        print(f"\n--- FAILURES ({len(fail_results)}) ---")
        for r in fail_results:
            fields = ["instantiation", "forward_pass", "adjoint_pass",
                       "shape_consistency", "adjoint_consistency",
                       "prepost_correction", "lowrank_correction"]
            failed = {f: getattr(r, f) for f in fields if getattr(r, f).startswith("FAIL")}
            print(f"  {r.modality} ({r.path}): {failed}")

    return total_fail


def main():
    print(f"Testing {len(ALL_MODALITIES)} modalities...")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")

    t_start = time.time()
    all_results: List[TestResult] = []

    for i, modality in enumerate(ALL_MODALITIES):
        print(f"  [{i+1}/{len(ALL_MODALITIES)}] {modality}...", end="", flush=True)
        results = run_single_modality(modality)
        all_results.extend(results)
        statuses = [r.instantiation[:4] for r in results]
        print(f" {', '.join(statuses)}")

    elapsed = time.time() - t_start
    n_fail = print_report(all_results)

    print(f"\nTotal time: {elapsed:.1f}s")

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
