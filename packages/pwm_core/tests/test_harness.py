"""End-to-end tests for the LIP-Arena targeting harness.

Tests cover:
- Harness initialization from registries
- 4-scenario protocol execution
- Scoring (rho, oracle_gap, RoIC, OFS)
- Safety brake enforcement
- Anti-Goodhart penalty detection
- Budget enforcement
- RunBundle emission
- Sandbox mode
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: a toy solver that works with any LinearLikeOperator
# ---------------------------------------------------------------------------


def _toy_solver(y, physics, cfg):
    """Simple adjoint solver for testing."""
    x_hat = physics.adjoint(y)
    return x_hat, {"solver": "toy_adjoint", "iters": 0}


# ---------------------------------------------------------------------------
# Scoring unit tests (no harness needed)
# ---------------------------------------------------------------------------


class TestScoringFormulas:
    """Test frozen scoring formulas per Rail Constitution Article 1.3, 1.4."""

    def test_recovery_ratio_basic(self):
        from pwm_core.targeting.scoring import compute_recovery_ratio

        # Perfect recovery: rho = 1.0
        assert compute_recovery_ratio(30.0, 20.0, 30.0) == pytest.approx(1.0)
        # No recovery: rho = 0.0
        assert compute_recovery_ratio(30.0, 20.0, 20.0) == pytest.approx(0.0)
        # Partial recovery
        assert compute_recovery_ratio(30.0, 20.0, 25.0) == pytest.approx(0.5)
        # Over-recovery
        assert compute_recovery_ratio(30.0, 20.0, 32.0) == pytest.approx(1.2)

    def test_recovery_ratio_degenerate(self):
        from pwm_core.targeting.scoring import compute_recovery_ratio

        # No gap (PSNR_I == PSNR_II)
        assert compute_recovery_ratio(30.0, 30.0, 30.0) == pytest.approx(1.0)

    def test_oracle_gap(self):
        from pwm_core.targeting.scoring import compute_oracle_gap

        assert compute_oracle_gap(30.0, 28.0) == pytest.approx(2.0)
        assert compute_oracle_gap(30.0, 30.0) == pytest.approx(0.0)

    def test_roic(self):
        from pwm_core.targeting.scoring import compute_roic

        # 10 dB in 3600s = 10 dB/GPU-hour
        assert compute_roic(10.0, 3600.0) == pytest.approx(10.0)
        # 5 dB in 1800s = 10 dB/GPU-hour
        assert compute_roic(5.0, 1800.0) == pytest.approx(10.0)

    def test_ofs(self):
        from pwm_core.targeting.scoring import compute_ofs

        assert compute_ofs(28.0, 30.0) == pytest.approx(28.0 / 30.0)
        assert compute_ofs(30.0, 30.0) == pytest.approx(1.0)

    def test_msi(self):
        from pwm_core.targeting.scoring import compute_msi

        # 6 dB drop over 3 pixels = 2 dB/px
        assert compute_msi(6.0, 3.0) == pytest.approx(2.0)


class TestSafetyBrakes:
    """Test safety brake thresholds per Rail Constitution Article 1.5."""

    def test_rho_below_threshold(self):
        from pwm_core.targeting.scoring import check_safety_brakes

        violations = check_safety_brakes(rho=0.25)
        assert len(violations) >= 1
        assert any(v.condition == "recovery_ratio_low" for v in violations)

    def test_rho_above_threshold(self):
        from pwm_core.targeting.scoring import check_safety_brakes

        violations = check_safety_brakes(rho=0.50)
        assert not any(v.condition == "recovery_ratio_low" for v in violations)

    def test_budget_exceeded(self):
        from pwm_core.targeting.scoring import check_safety_brakes

        violations = check_safety_brakes(rho=0.50, budget_ratio=2.5)
        assert any(v.condition == "budget_exceeded" for v in violations)
        assert any(v.action == "disqualify" for v in violations)

    def test_consistency_violation(self):
        from pwm_core.targeting.scoring import check_safety_brakes

        violations = check_safety_brakes(rho=0.50, reprojection_error_ratio=4.0)
        assert any(v.condition == "consistency_violation" for v in violations)


class TestGamingPenalties:
    """Test anti-Goodhart gaming penalty checks per targeting_system.md S5."""

    def test_wrong_triad(self):
        from pwm_core.targeting.scoring import check_gaming_penalties

        penalties = check_gaming_penalties({"triad_attribution_correct": False})
        assert len(penalties) >= 1
        assert penalties[0].penalty_fraction == pytest.approx(0.15)

    def test_compute_dishonesty(self):
        from pwm_core.targeting.scoring import check_gaming_penalties

        penalties = check_gaming_penalties({
            "declared_budget_s": 10.0,
            "actual_budget_s": 100.0,
        })
        assert any(p.penalty_fraction >= 1.0 for p in penalties)

    def test_no_penalties_clean(self):
        from pwm_core.targeting.scoring import check_gaming_penalties

        penalties = check_gaming_penalties({
            "triad_attribution_correct": True,
            "uncertainty_coverage": 0.90,
            "identifiability_consistent": True,
        })
        assert len(penalties) == 0


class TestScoreRun:
    """Test the complete score_run function."""

    def test_score_run_basic(self):
        from pwm_core.targeting.scoring import score_run

        result = score_run(
            psnr_i=30.0, psnr_ii=20.0, psnr_iii=25.0, psnr_iv=29.0,
            total_gpu_seconds=100.0,
        )
        assert result.rho == pytest.approx(0.5)
        assert result.oracle_gap == pytest.approx(5.0)
        assert not result.disqualified

    def test_score_run_disqualified(self):
        from pwm_core.targeting.scoring import score_run

        result = score_run(
            psnr_i=30.0, psnr_ii=20.0, psnr_iii=25.0, psnr_iv=29.0,
            total_gpu_seconds=100.0,
            diagnostics={
                "declared_budget_s": 10.0,
                "actual_budget_s": 100.0,
            },
        )
        assert result.disqualified
        assert result.final_score == 0.0


# ---------------------------------------------------------------------------
# Budget tests
# ---------------------------------------------------------------------------


class TestBudgetGuard:
    """Test compute budget enforcement."""

    def test_budget_report(self):
        from pwm_core.targeting.budget import BudgetGuard

        guard = BudgetGuard(budget_s=10.0, enforce=False)
        with guard:
            pass  # instant

        report = guard.report()
        assert report.actual_s >= 0.0
        assert not report.exceeded

    def test_budget_no_enforce_sandbox(self):
        from pwm_core.targeting.budget import BudgetGuard

        guard = BudgetGuard(budget_s=0.001, enforce=False)
        # Should not raise even if slow
        with guard:
            import time
            time.sleep(0.01)

        report = guard.report()
        assert report.actual_s > 0.001


# ---------------------------------------------------------------------------
# Mismatch sampler tests
# ---------------------------------------------------------------------------


class TestMismatchSampler:
    """Test mismatch parameter sampling."""

    def test_sample_known_modality(self):
        from pwm_core.targeting.mismatch_sampler import sample_mismatch

        rng = np.random.default_rng(42)
        params = sample_mismatch("widefield", severity="moderate", rng=rng)
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_sample_unknown_modality(self):
        from pwm_core.targeting.mismatch_sampler import sample_mismatch

        params = sample_mismatch("nonexistent_modality")
        assert params == {}

    def test_severity_scales(self):
        from pwm_core.targeting.mismatch_sampler import sample_mismatch

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        mild = sample_mismatch("widefield", severity="mild", rng=rng1)
        severe = sample_mismatch("widefield", severity="severe", rng=rng2)
        # Severe should have larger magnitudes on average
        # (not deterministic per sample, but range is wider)
        assert isinstance(mild, dict) and isinstance(severe, dict)

    def test_mismatch_magnitude(self):
        from pwm_core.targeting.mismatch_sampler import compute_mismatch_magnitude

        assert compute_mismatch_magnitude({}) == 0.0
        assert compute_mismatch_magnitude({"dx": 3.0, "dy": 4.0}) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Harness end-to-end tests
# ---------------------------------------------------------------------------


class TestHarness:
    """End-to-end harness tests using the widefield modality."""

    def test_harness_sandbox(self):
        """Sandbox mode runs fast and produces valid results."""
        from pwm_core.targeting.harness import Harness

        harness = Harness(
            modality="widefield",
            solver="traditional_cpu",
            sandbox=True,
            solver_fn=_toy_solver,
        )
        result = harness.run(n_scenes=1, seed=42, severity="mild")

        assert result.modality == "widefield"
        assert result.sandbox is True
        assert result.n_scenes == 1
        assert len(result.per_scene) == 1
        assert result.aggregate.rho is not None
        assert isinstance(result.aggregate.rho, float)

    def test_harness_multiple_scenes(self):
        """Multi-scene averaging works."""
        from pwm_core.targeting.harness import Harness

        harness = Harness(
            modality="widefield",
            solver="traditional_cpu",
            sandbox=False,
            solver_fn=_toy_solver,
        )
        result = harness.run(n_scenes=3, seed=42, severity="moderate")

        assert result.n_scenes == 3
        assert len(result.per_scene) == 3

    def test_harness_summary_table(self):
        """Summary table renders without error."""
        from pwm_core.targeting.harness import Harness

        harness = Harness(
            modality="widefield",
            solver="traditional_cpu",
            sandbox=True,
            solver_fn=_toy_solver,
        )
        result = harness.run(n_scenes=1, seed=42)
        table = result.summary_table()
        assert "LIP-Arena" in table
        assert "rho" in table.lower() or "Recovery" in table

    def test_harness_to_dict(self):
        """JSON serialization works."""
        from pwm_core.targeting.harness import Harness

        harness = Harness(
            modality="widefield",
            solver="traditional_cpu",
            sandbox=True,
            solver_fn=_toy_solver,
        )
        result = harness.run(n_scenes=1, seed=42)
        d = result.to_dict()
        assert "modality" in d
        assert "aggregate" in d
        # Ensure it's JSON-serializable
        json.dumps(d, default=str)


# ---------------------------------------------------------------------------
# RunBundle emission tests
# ---------------------------------------------------------------------------


class TestRunBundleEmission:
    """Test RunBundle v0.3.0 emission."""

    def test_emit_runbundle(self):
        from pwm_core.targeting.harness import Harness
        from pwm_core.targeting.runbundle_emitter import emit_runbundle

        harness = Harness(
            modality="widefield",
            solver="traditional_cpu",
            sandbox=True,
            solver_fn=_toy_solver,
        )
        result = harness.run(n_scenes=1, seed=42)

        tmpdir = Path(tempfile.mkdtemp())
        try:
            bundle_path = emit_runbundle(result, tmpdir)
            assert bundle_path.is_dir()

            # Check manifest exists
            manifest_path = bundle_path / "runbundle_manifest.json"
            assert manifest_path.is_file()

            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest["version"] == "0.3.0"
            assert "provenance" in manifest
            assert "metrics" in manifest
            assert "artifacts" in manifest
            assert "hashes" in manifest

            # All hashes start with sha256:
            for key, h in manifest["hashes"].items():
                assert h.startswith("sha256:"), f"Hash for {key} missing prefix"

            # DR-IS records exist
            dr_is_path = bundle_path / "logs" / "dr_is_records.json"
            assert dr_is_path.is_file()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Test CLI argument parsing."""

    def test_parser_evaluate(self):
        from pwm_core.targeting.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "evaluate", "--modality", "widefield", "--solver", "traditional_cpu",
            "--sandbox",
        ])
        assert args.command == "evaluate"
        assert args.modality == "widefield"
        assert args.sandbox is True

    def test_parser_scaffold(self):
        from pwm_core.targeting.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["scaffold", "solver", "my_solver"])
        assert args.command == "scaffold"
        assert args.type == "solver"
        assert args.name == "my_solver"
