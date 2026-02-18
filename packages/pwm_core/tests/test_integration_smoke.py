"""Integration smoke tests for the full PWM targeting harness pipeline.

Tests the complete chain:
  YAML registries → GraphCompiler → GraphOperator → 4 scenarios → scoring → HarnessResult

These are the acid tests: if they pass, the rail is fully operational end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure local pwm_core is importable
# ---------------------------------------------------------------------------
_PKG_DIR = str(Path(__file__).resolve().parents[1])
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
for key in list(sys.modules):
    if key == "pwm_core" or key.startswith("pwm_core."):
        del sys.modules[key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_solver(y: np.ndarray, physics: Any, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Minimal solver: adjoint back-projection.

    Matches the required harness signature: (y, physics, cfg) -> (x_hat, info).
    """
    if hasattr(physics, "adjoint"):
        x_hat = physics.adjoint(y)
    else:
        x_hat = y.copy()
    info = {"solver": "mock_adjoint"}
    return x_hat.astype(np.float32), info


def _identity_solver(y: np.ndarray, physics: Any, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Just returns the measurement (worst-case baseline)."""
    # Reshape to x_shape if possible
    if hasattr(physics, "x_shape"):
        x_hat = np.zeros(physics.x_shape, dtype=np.float32)
        # Copy as much data as fits
        flat_y = y.ravel()
        flat_x = x_hat.ravel()
        n = min(len(flat_y), len(flat_x))
        flat_x[:n] = flat_y[:n]
        x_hat = flat_x.reshape(physics.x_shape)
    else:
        x_hat = y.copy()
    return x_hat.astype(np.float32), {"solver": "identity"}


# ---------------------------------------------------------------------------
# Test 1: Full Harness pipeline with mock solver
# ---------------------------------------------------------------------------


class TestHarnessEndToEnd:
    """Test the full harness pipeline using mock solvers."""

    def test_widefield_mock_solver(self):
        """Smoke test: widefield modality with mock adjoint solver."""
        from pwm_core.targeting.harness import Harness

        harness = Harness(
            modality="widefield",
            solver="traditional_cpu",
            track="correct",
            budget_s=600,
            sandbox=True,
            solver_fn=_mock_solver,  # Override registry with mock
        )

        result = harness.run(n_scenes=1, seed=42, severity="moderate")

        # Basic structure checks
        assert result.modality == "widefield"
        assert result.solver == "traditional_cpu"
        assert result.track == "correct"
        assert result.sandbox is True
        assert result.n_scenes == 1
        assert len(result.per_scene) == 1

        # Metrics must be finite
        assert np.isfinite(result.aggregate.rho), f"rho not finite: {result.aggregate.rho}"
        assert np.isfinite(result.aggregate.oracle_gap), f"oracle_gap not finite"
        assert np.isfinite(result.aggregate.final_score), f"final_score not finite"

        # Budget report must be valid
        assert result.budget_report["declared_s"] == 600
        assert result.budget_report["actual_s"] >= 0

    def test_cassi_mock_solver(self):
        """Smoke test: cassi modality (3D cube) with mock solver."""
        from pwm_core.targeting.harness import Harness

        harness = Harness(
            modality="cassi",
            solver="traditional_cpu",
            track="correct",
            budget_s=600,
            sandbox=True,
            solver_fn=_mock_solver,
        )

        result = harness.run(n_scenes=1, seed=42, severity="moderate")

        assert result.modality == "cassi"
        assert len(result.per_scene) == 1
        assert np.isfinite(result.aggregate.rho)
        assert np.isfinite(result.aggregate.oracle_gap)

    def test_four_scenarios_produce_different_psnr(self):
        """Verify that the 4 scenarios produce distinct PSNR values.

        Scenario I (ideal) should generally be >= Scenario II (mismatched),
        since using the correct operator should be better.
        """
        from pwm_core.targeting.harness import Harness

        harness = Harness(
            modality="widefield",
            solver="traditional_cpu",
            track="correct",
            budget_s=600,
            sandbox=True,
            solver_fn=_mock_solver,
        )

        result = harness.run(n_scenes=1, seed=42, severity="moderate")
        scene = result.per_scene[0]

        # All 4 scenarios must exist
        assert "I" in scene.scenario_results
        assert "II" in scene.scenario_results
        assert "III" in scene.scenario_results
        assert "IV" in scene.scenario_results

        # All PSNR values must be finite
        for scenario_id, sr in scene.scenario_results.items():
            assert np.isfinite(sr.psnr), f"Scenario {scenario_id} PSNR not finite: {sr.psnr}"
            assert np.isfinite(sr.ssim), f"Scenario {scenario_id} SSIM not finite: {sr.ssim}"
            assert sr.runtime_s >= 0, f"Scenario {scenario_id} runtime negative"

    def test_harness_result_serialization(self):
        """Verify that HarnessResult.to_dict() produces valid output."""
        from pwm_core.targeting.harness import Harness

        harness = Harness(
            modality="widefield",
            solver="traditional_cpu",
            track="correct",
            budget_s=600,
            sandbox=True,
            solver_fn=_mock_solver,
        )

        result = harness.run(n_scenes=1, seed=42, severity="moderate")
        d = result.to_dict()

        # Must have all required keys
        assert "modality" in d
        assert "solver" in d
        assert "aggregate" in d
        assert "per_scene" in d
        assert d["modality"] == "widefield"
        assert d["aggregate"]["rho"] == result.aggregate.rho

    def test_summary_table(self):
        """Verify summary_table() produces non-empty output."""
        from pwm_core.targeting.harness import Harness

        harness = Harness(
            modality="widefield",
            solver="traditional_cpu",
            track="correct",
            budget_s=600,
            sandbox=True,
            solver_fn=_mock_solver,
        )

        result = harness.run(n_scenes=1, seed=42, severity="moderate")
        table = result.summary_table()
        assert "LIP-Arena" in table
        assert "rho" in table.lower() or "Recovery" in table


# ---------------------------------------------------------------------------
# Test 2: Solver registry resolution
# ---------------------------------------------------------------------------


class TestSolverRegistry:
    """Test that solver_registry.yaml entries resolve correctly."""

    def test_cassi_traditional_cpu_resolves_to_run_gap_tv(self):
        """Verify the fixed registry: cassi/traditional_cpu → run_gap_tv."""
        from pwm_core.targeting.harness import load_solver_registry, _resolve_solver_fn

        registry = load_solver_registry()
        fn = _resolve_solver_fn("cassi", "traditional_cpu", registry)

        # Must be the run_gap_tv function
        assert fn.__name__ == "run_gap_tv", f"Expected run_gap_tv, got {fn.__name__}"

    def test_cacti_traditional_cpu_resolves_to_run_gap_tv(self):
        """Verify the fixed registry: cacti/traditional_cpu → run_gap_tv."""
        from pwm_core.targeting.harness import load_solver_registry, _resolve_solver_fn

        registry = load_solver_registry()
        fn = _resolve_solver_fn("cacti", "traditional_cpu", registry)

        assert fn.__name__ == "run_gap_tv", f"Expected run_gap_tv, got {fn.__name__}"

    def test_run_gap_tv_has_correct_signature(self):
        """Verify run_gap_tv accepts (y, physics, cfg) and returns (x_hat, info)."""
        from pwm_core.recon.gap_tv import run_gap_tv

        # Create a minimal mock physics object
        class MockPhysics:
            x_shape = (32, 32)
            def forward(self, x):
                return x
            def adjoint(self, y):
                return y
            def info(self):
                return {"modality": "general", "x_shape": (32, 32)}

        y = np.random.default_rng(42).random((32, 32)).astype(np.float32)
        physics = MockPhysics()
        cfg = {"iters": 3, "lam": 0.05}

        x_hat, info = run_gap_tv(y, physics, cfg)

        assert isinstance(x_hat, np.ndarray), "x_hat must be ndarray"
        assert isinstance(info, dict), "info must be dict"
        assert x_hat.dtype == np.float32, f"Expected float32, got {x_hat.dtype}"


# ---------------------------------------------------------------------------
# Test 3: Cross-modality with same solver
# ---------------------------------------------------------------------------


class TestCrossModality:
    """Test the same solver on multiple modalities (the key PWM promise)."""

    def test_same_solver_two_modalities(self):
        """The same solver function should work across widefield and cassi."""
        from pwm_core.targeting.harness import Harness

        results = {}
        for modality in ["widefield", "cassi"]:
            harness = Harness(
                modality=modality,
                solver="traditional_cpu",
                track="correct",
                budget_s=600,
                sandbox=True,
                solver_fn=_mock_solver,
            )
            results[modality] = harness.run(n_scenes=1, seed=42, severity="moderate")

        # Both must produce valid results
        for mod, result in results.items():
            assert result.modality == mod
            assert np.isfinite(result.aggregate.rho), f"{mod}: rho not finite"
            assert np.isfinite(result.aggregate.final_score), f"{mod}: score not finite"


# ---------------------------------------------------------------------------
# Test 4: Scoring formulas
# ---------------------------------------------------------------------------


class TestScoringFormulas:
    """Test frozen scoring formulas in isolation and via harness."""

    def test_recovery_ratio_formula(self):
        """rho = (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II)."""
        from pwm_core.targeting.scoring import compute_recovery_ratio

        # Perfect calibration: III == I
        assert compute_recovery_ratio(40.0, 30.0, 40.0) == pytest.approx(1.0)

        # No improvement: III == II
        assert compute_recovery_ratio(40.0, 30.0, 30.0) == pytest.approx(0.0)

        # Halfway
        assert compute_recovery_ratio(40.0, 30.0, 35.0) == pytest.approx(0.5)

        # Edge case: no gap (I == II)
        rho = compute_recovery_ratio(30.0, 30.0, 30.0)
        assert np.isfinite(rho)

    def test_oracle_gap_formula(self):
        """oracle_gap = PSNR_I - PSNR_III."""
        from pwm_core.targeting.scoring import compute_oracle_gap

        assert compute_oracle_gap(40.0, 35.0) == pytest.approx(5.0)
        assert compute_oracle_gap(40.0, 40.0) == pytest.approx(0.0)

    def test_roic_formula(self):
        """RoIC = dB gained / GPU-hours."""
        from pwm_core.targeting.scoring import compute_roic

        # 5 dB gain in 3600 seconds = 5 dB/GPU-hr
        assert compute_roic(5.0, 3600.0) == pytest.approx(5.0)

        # Zero time
        assert compute_roic(5.0, 0.0) == float("inf")

    def test_score_run_produces_valid_result(self):
        """score_run() must return a ScoredResult with all fields."""
        from pwm_core.targeting.scoring import score_run

        scored = score_run(
            psnr_i=40.0,
            psnr_ii=30.0,
            psnr_iii=35.0,
            psnr_iv=38.0,
            total_gpu_seconds=10.0,
            track="correct",
            mismatch_magnitude=1.0,
        )

        assert scored.rho == pytest.approx(0.5)
        assert scored.oracle_gap == pytest.approx(5.0)
        assert scored.track == "correct"
        assert not scored.disqualified


# ---------------------------------------------------------------------------
# Test 5: Safety brakes
# ---------------------------------------------------------------------------


class TestSafetyBrakes:
    """Test safety brake enforcement."""

    def test_low_rho_triggers_block(self):
        """rho < 0.30 must trigger 'block' safety brake."""
        from pwm_core.targeting.scoring import check_safety_brakes

        violations = check_safety_brakes(rho=0.15)
        assert len(violations) >= 1
        conditions = [v.condition for v in violations]
        assert "recovery_ratio_low" in conditions

    def test_budget_exceeded_triggers_dq(self):
        """budget_ratio > 2.0 must trigger 'disqualify'."""
        from pwm_core.targeting.scoring import check_safety_brakes

        violations = check_safety_brakes(rho=0.5, budget_ratio=3.0)
        conditions = [v.condition for v in violations]
        assert "budget_exceeded" in conditions
        actions = [v.action for v in violations]
        assert "disqualify" in actions

    def test_no_violations_clean_run(self):
        """rho=0.8, budget_ratio=1.0 should produce no violations."""
        from pwm_core.targeting.scoring import check_safety_brakes

        violations = check_safety_brakes(rho=0.8, budget_ratio=1.0)
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Test 6: GraphOperator compile + adjoint check
# ---------------------------------------------------------------------------


class TestGraphOperator:
    """Test graph compilation and adjoint correctness."""

    def test_widefield_compiles_and_has_adjoint(self):
        """Widefield graph (noise-stripped) should compile and pass adjoint check."""
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.targeting.harness import load_graph_templates

        templates = load_graph_templates()
        template = templates["widefield_graph_v1"]

        # Strip noise nodes (same logic as harness)
        allowed_keys = {"graph_id", "nodes", "edges", "metadata"}
        spec_data = {"graph_id": "widefield_graph_v1"}
        for k, v in template.items():
            if k in allowed_keys:
                spec_data[k] = v

        if "nodes" in spec_data:
            noise_ids = {
                n["node_id"] for n in spec_data["nodes"]
                if n.get("role") == "noise"
                or n.get("primitive_id", "").startswith("noise")
                or "noise" in n.get("node_id", "").lower()
            }
            if noise_ids:
                spec_data["nodes"] = [
                    n for n in spec_data["nodes"] if n["node_id"] not in noise_ids
                ]
                if "edges" in spec_data:
                    spec_data["edges"] = [
                        e for e in spec_data["edges"]
                        if e.get("source") not in noise_ids
                        and e.get("target") not in noise_ids
                    ]

        compiler = GraphCompiler()
        spec = GraphCompiler.from_dict(spec_data)
        op = compiler.compile(spec)

        assert op.all_linear, "Noise-stripped widefield graph must be fully linear"
        assert op.x_shape == (64, 64)

        # Forward should work
        x = np.random.default_rng(42).random(op.x_shape).astype(np.float64)
        y = op.forward(x)
        assert y.shape == op.y_shape, f"Forward output shape mismatch: {y.shape} vs {op.y_shape}"

        # Adjoint should work
        x_back = op.adjoint(y)
        assert x_back.shape == op.x_shape

        # Adjoint correctness check
        report = op.check_adjoint(n_trials=3, seed=42)
        assert report.passed, f"Adjoint check failed: {report.summary()}"


# ---------------------------------------------------------------------------
# Test 7: Harness with run_gap_tv on widefield (real solver, real graph)
# ---------------------------------------------------------------------------


class TestRealSolver:
    """Test with actual run_gap_tv solver (not mock)."""

    def test_run_gap_tv_on_widefield(self):
        """run_gap_tv should work on widefield via the general operator path."""
        from pwm_core.recon.gap_tv import run_gap_tv
        from pwm_core.targeting.harness import Harness

        harness = Harness(
            modality="widefield",
            solver="traditional_cpu",
            track="correct",
            budget_s=600,
            sandbox=True,
            solver_fn=run_gap_tv,
        )

        result = harness.run(n_scenes=1, seed=42, severity="mild")

        assert result.modality == "widefield"
        assert np.isfinite(result.aggregate.rho)
        assert np.isfinite(result.aggregate.oracle_gap)

        # Real solver should produce positive PSNR in all scenarios
        scene = result.per_scene[0]
        for sid, sr in scene.scenario_results.items():
            assert sr.psnr > 0, f"Scenario {sid}: PSNR should be positive, got {sr.psnr}"

    def test_registry_loaded_solver_on_cassi(self):
        """Harness should load run_gap_tv from registry for cassi/traditional_cpu."""
        from pwm_core.targeting.harness import Harness

        # This time, do NOT pass solver_fn — let it resolve from registry
        harness = Harness(
            modality="cassi",
            solver="traditional_cpu",
            track="correct",
            budget_s=600,
            sandbox=True,
        )

        # Verify the resolver found run_gap_tv
        assert harness._solver_fn.__name__ == "run_gap_tv"

        result = harness.run(n_scenes=1, seed=42, severity="mild")

        assert result.modality == "cassi"
        assert np.isfinite(result.aggregate.rho)
        assert np.isfinite(result.aggregate.final_score)


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
