"""End-to-end integration tests for the LIP-Arena targeting harness.

Tests that ``Harness.run()`` completes without error for each of the
7 validated modalities in sandbox mode, producing valid scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure pwm_core is importable
_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo / "packages" / "pwm_core"))

from pwm_core.targeting.harness import DecisionRecord, Harness, HarnessResult
from pwm_core.targeting.scoring import (
    GateAttribution,
    MaturityLevel,
    compute_maturity_level,
    infer_gate_attribution,
)

VALIDATED_MODALITIES = [
    "cassi", "cacti", "spc", "ct", "mri", "ptychography", "lensless",
]


@pytest.fixture(params=VALIDATED_MODALITIES)
def modality(request):
    return request.param


def _run_sandbox(modality: str) -> HarnessResult:
    """Run the harness in sandbox mode for a single modality."""
    harness = Harness(
        modality=modality,
        solver="traditional_cpu",
        track="correct",
        budget_s=120,
        sandbox=True,
    )
    return harness.run(n_scenes=1, seed=42, severity="mild")


class TestHarnessE2E:
    """End-to-end harness tests for all validated modalities."""

    def test_harness_runs_without_error(self, modality):
        """Harness.run() completes without exceptions."""
        result = _run_sandbox(modality)
        assert isinstance(result, HarnessResult)

    def test_harness_produces_valid_rho(self, modality):
        """Recovery ratio is a finite number (can be nan if no gap)."""
        result = _run_sandbox(modality)
        rho = result.aggregate.rho
        # rho can be nan when PSNR_I == PSNR_II, but must not be None
        assert rho is not None

    def test_harness_not_disqualified(self, modality):
        """Sandbox runs should not trigger disqualification."""
        result = _run_sandbox(modality)
        assert not result.aggregate.disqualified, (
            f"Disqualified: {result.aggregate.disqualification_reason}"
        )

    def test_harness_has_per_scene_results(self, modality):
        """Sandbox runs 1 scene and populates per_scene."""
        result = _run_sandbox(modality)
        assert len(result.per_scene) == 1
        scene = result.per_scene[0]
        assert "I" in scene.scenario_results
        assert "II" in scene.scenario_results
        assert "III" in scene.scenario_results
        assert "IV" in scene.scenario_results

    def test_harness_timing_positive(self, modality):
        """Total timing should be positive."""
        result = _run_sandbox(modality)
        assert result.timing_s > 0

    def test_harness_to_dict(self, modality):
        """to_dict() returns a valid dict with expected keys."""
        result = _run_sandbox(modality)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["modality"] == modality
        assert "aggregate" in d
        assert "per_scene" in d

    def test_harness_gate_attribution(self, modality):
        """Gate attribution is computed and has valid structure."""
        result = _run_sandbox(modality)

        # Aggregate gate attribution must be present
        assert result.gate_attribution is not None, "gate_attribution missing on HarnessResult"
        ga = result.gate_attribution

        # Fractions are non-negative
        assert ga.gate_1_sampling >= 0.0
        assert ga.gate_2_noise >= 0.0
        assert ga.gate_3_mismatch >= 0.0

        # Fractions sum to ~1.0
        total = ga.gate_1_sampling + ga.gate_2_noise + ga.gate_3_mismatch
        assert abs(total - 1.0) < 1e-3, f"Gate fractions sum to {total:.4f}, expected 1.0"

        # Dominant gate is one of the three valid gates
        assert ga.dominant_gate in (
            "gate_1_sampling", "gate_2_noise", "gate_3_mismatch"
        )

        # Confidence is in [0, 1]
        assert 0.0 <= ga.confidence <= 1.0

        # Recommended action is a non-empty string
        assert isinstance(ga.recommended_action, str) and len(ga.recommended_action) > 0

        # Per-scene gate attribution is also present
        assert result.per_scene[0].gate_attribution is not None

        # to_dict() includes gate_attribution
        d = result.to_dict()
        assert "gate_attribution" in d
        assert d["gate_attribution"] is not None
        assert "dominant_gate" in d["gate_attribution"]


class TestDatasets:
    """Verify that generated LIP-Arena datasets exist and are valid."""

    DATASET_DIR = _repo / "datasets" / "lip_arena"

    @pytest.mark.parametrize("modality", VALIDATED_MODALITIES)
    def test_dataset_exists(self, modality):
        """Each validated modality has x_gt.npy, y.npy, metadata.json."""
        mod_dir = self.DATASET_DIR / modality
        assert mod_dir.exists(), f"Missing dataset dir: {mod_dir}"
        assert (mod_dir / "x_gt.npy").exists()
        assert (mod_dir / "y.npy").exists()
        assert (mod_dir / "metadata.json").exists()

    @pytest.mark.parametrize("modality", VALIDATED_MODALITIES)
    def test_dataset_loadable(self, modality):
        """Datasets can be loaded and have consistent shapes."""
        import json

        mod_dir = self.DATASET_DIR / modality
        x_gt = np.load(mod_dir / "x_gt.npy")
        y = np.load(mod_dir / "y.npy")

        with open(mod_dir / "metadata.json") as f:
            meta = json.load(f)

        assert list(x_gt.shape) == meta["x_shape"]
        assert list(y.shape) == meta["y_shape"]
        assert x_gt.size > 0
        assert y.size > 0


class TestGateAttribution:
    """Unit tests for Triad gate attribution logic (infer_gate_attribution)."""

    def _check_valid(self, ga: GateAttribution) -> None:
        """Shared validity checks for a GateAttribution."""
        fracs = [ga.gate_1_sampling, ga.gate_2_noise, ga.gate_3_mismatch]
        assert all(f >= 0.0 for f in fracs), f"Negative fraction: {fracs}"
        total = sum(fracs)
        assert abs(total - 1.0) < 1e-3, f"Fractions sum to {total:.4f}"
        assert ga.dominant_gate in (
            "gate_1_sampling", "gate_2_noise", "gate_3_mismatch"
        )
        assert 0.0 <= ga.confidence <= 1.0

    def test_gate3_dominant_when_calibration_recovers(self):
        """When calibration (III) recovers most of the II→I gap, Gate 3 is dominant."""
        # PSNR_I=30, PSNR_II=15, PSNR_III=28 → Gate 3 recovers 13/15 = 87%
        ga = infer_gate_attribution(30.0, 15.0, 28.0, 30.0, mismatch_magnitude=1.5)
        self._check_valid(ga)
        assert ga.dominant_gate == "gate_3_mismatch", (
            f"Expected gate_3_mismatch, got {ga.dominant_gate} "
            f"(g3={ga.gate_3_mismatch:.3f})"
        )
        assert ga.gate_3_mismatch > 0.6

    def test_gate12_dominant_when_calibration_does_not_help(self):
        """When calibration barely helps, Gates 1/2 dominate."""
        # PSNR_I=30, PSNR_II=15, PSNR_III=16 → calibration gained only 1/15 = 7%
        ga = infer_gate_attribution(30.0, 15.0, 16.0, 30.0, mismatch_magnitude=0.5)
        self._check_valid(ga)
        # Gate 3 should be small; Gates 1+2 dominant
        assert ga.gate_3_mismatch < 0.2

    def test_near_ideal_performance_low_confidence(self):
        """When all PSNRs are equal (near-ideal), confidence should be low."""
        ga = infer_gate_attribution(30.0, 30.0, 30.0, 30.0, mismatch_magnitude=0.0)
        self._check_valid(ga)
        assert ga.confidence < 0.5, (
            f"Expected low confidence for trivial case, got {ga.confidence}"
        )

    def test_high_psnr_with_large_oracle_gap_prefers_gate1(self):
        """High PSNR_I with large oracle gap suggests sampling bottleneck (Gate 1)."""
        # PSNR_I=35 (good SNR), PSNR_III=28 (7 dB oracle gap), minimal Gate 3 gain
        ga = infer_gate_attribution(35.0, 25.0, 26.0, 35.0, mismatch_magnitude=0.3)
        self._check_valid(ga)
        # Residual (oracle_gap=9 dB) should favor Gate 1 over Gate 2
        assert ga.gate_1_sampling > ga.gate_2_noise

    def test_low_psnr_prefers_gate2_noise(self):
        """Low absolute PSNR_I suggests noise is the primary limit (Gate 2)."""
        # PSNR_I=12 (poor SNR), calibration doesn't help much
        ga = infer_gate_attribution(12.0, 5.0, 6.0, 12.0, mismatch_magnitude=0.3)
        self._check_valid(ga)
        # Residual should favor Gate 2 over Gate 1
        assert ga.gate_2_noise > ga.gate_1_sampling

    def test_recommended_action_matches_dominant_gate(self):
        """Recommended action must correspond to the dominant gate."""
        ga = infer_gate_attribution(30.0, 15.0, 28.0, 30.0, mismatch_magnitude=1.5)
        self._check_valid(ga)
        if ga.dominant_gate == "gate_3_mismatch":
            assert "operator" in ga.recommended_action.lower() or "calibrat" in ga.recommended_action.lower()
        elif ga.dominant_gate == "gate_1_sampling":
            assert "sampl" in ga.recommended_action.lower() or "measurement" in ga.recommended_action.lower()
        elif ga.dominant_gate == "gate_2_noise":
            assert "noise" in ga.recommended_action.lower()

    def test_to_dict_has_required_keys(self):
        """to_dict() includes all expected fields."""
        ga = infer_gate_attribution(30.0, 15.0, 28.0, 30.0, mismatch_magnitude=1.5)
        d = ga.to_dict()
        for key in (
            "gate_1_sampling", "gate_2_noise", "gate_3_mismatch",
            "dominant_gate", "confidence", "recommended_action", "evidence",
        ):
            assert key in d, f"Missing key '{key}' in to_dict()"


class TestMaturityLevel:
    """Unit tests for the SolveEverything L0-L5 maturation curve."""

    def test_l0_for_zero_rho(self):
        """rho=0.0 → L0 (Ill-Posed)."""
        ml = compute_maturity_level(0.0)
        assert ml.level == 0
        assert ml.label == "L0"
        assert ml.name == "Ill-Posed"

    def test_l1_boundary(self):
        """rho=0.10 exactly → L1 (Measurable)."""
        ml = compute_maturity_level(0.10)
        assert ml.level == 1
        assert ml.label == "L1"

    def test_l2_mid(self):
        """rho=0.40 → L2 (Repeatable)."""
        ml = compute_maturity_level(0.40)
        assert ml.level == 2
        assert ml.label == "L2"
        assert ml.name == "Repeatable"

    def test_l3_boundary(self):
        """rho=0.50 exactly → L3 (Automated)."""
        ml = compute_maturity_level(0.50)
        assert ml.level == 3
        assert ml.label == "L3"
        assert ml.name == "Automated"

    def test_l4_mid(self):
        """rho=0.80 → L4 (Industrialized)."""
        ml = compute_maturity_level(0.80)
        assert ml.level == 4
        assert ml.label == "L4"
        assert ml.name == "Industrialized"

    def test_l5_boundary(self):
        """rho=0.90 exactly → L5 (Commoditized)."""
        ml = compute_maturity_level(0.90)
        assert ml.level == 5
        assert ml.label == "L5"
        assert ml.name == "Commoditized"

    def test_l5_above_threshold(self):
        """rho=1.0 → L5 (Commoditized)."""
        ml = compute_maturity_level(1.0)
        assert ml.level == 5

    def test_monotone_ordering(self):
        """Higher rho must give equal or higher maturity level."""
        rhos = [0.0, 0.05, 0.15, 0.35, 0.55, 0.75, 0.92, 1.0]
        levels = [compute_maturity_level(r).level for r in rhos]
        assert levels == sorted(levels), f"Non-monotone: {list(zip(rhos, levels))}"

    def test_to_dict_keys(self):
        """MaturityLevel.to_dict() has all required keys."""
        ml = compute_maturity_level(0.6)
        d = ml.to_dict()
        for key in ("level", "label", "name", "description", "rho_min"):
            assert key in d, f"Missing key '{key}'"

    def test_harness_result_has_maturity_level(self):
        """HarnessResult includes a valid maturity_level after run()."""
        result = _run_sandbox("cassi")
        assert result.maturity_level is not None
        assert isinstance(result.maturity_level, MaturityLevel)
        assert result.maturity_level.level in range(6)
        assert result.maturity_level.label.startswith("L")

    def test_harness_to_dict_includes_maturity(self):
        """to_dict() includes maturity_level block."""
        result = _run_sandbox("cassi")
        d = result.to_dict()
        assert "maturity_level" in d
        assert d["maturity_level"] is not None
        assert d["maturity_level"]["label"].startswith("L")

    def test_summary_table_shows_maturity(self):
        """summary_table() output includes the maturity label."""
        result = _run_sandbox("cassi")
        table = result.summary_table()
        assert "Maturity:" in table
        assert "L" in table  # at minimum L0-L5 label appears


class TestDecisionRecord:
    """Unit tests for DR-AIS DecisionRecord generation."""

    def test_harness_produces_decision_record(self):
        """HarnessResult includes a DecisionRecord after run()."""
        result = _run_sandbox("cassi")
        assert result.decision_record is not None
        assert isinstance(result.decision_record, DecisionRecord)

    def test_decision_record_fields(self):
        """DecisionRecord has correct modality, solver, and blinded_clear."""
        result = _run_sandbox("lensless")
        dr = result.decision_record
        assert dr.modality == "lensless"
        assert dr.solver == "traditional_cpu"
        assert dr.blinded_clear is True

    def test_decision_record_maturity_consistent(self):
        """DecisionRecord maturity_label must match HarnessResult.maturity_level."""
        result = _run_sandbox("spc")
        dr = result.decision_record
        ml = result.maturity_level
        assert ml is not None
        assert dr.maturity_label == ml.label
        assert dr.maturity_name == ml.name

    def test_decision_record_run_id_unique(self):
        """Each run produces a unique run_id."""
        r1 = _run_sandbox("cassi")
        r2 = _run_sandbox("cassi")
        assert r1.decision_record.run_id != r2.decision_record.run_id

    def test_decision_record_to_dict(self):
        """to_dict() includes decision_record with required keys."""
        result = _run_sandbox("cacti")
        d = result.to_dict()
        assert "decision_record" in d
        dr_dict = d["decision_record"]
        assert dr_dict is not None
        for key in (
            "run_id", "modality", "solver", "maturity_label", "maturity_name",
            "rho", "dominant_gate", "blinded_clear", "template_id", "severity",
            "n_scenes",
        ):
            assert key in dr_dict, f"Missing key '{key}' in decision_record dict"

    def test_decision_record_str(self):
        """__str__ produces a non-empty DR-AIS summary line."""
        result = _run_sandbox("ct")
        s = str(result.decision_record)
        assert "DR-AIS" in s
        assert result.decision_record.modality in s
        assert result.decision_record.maturity_label in s
