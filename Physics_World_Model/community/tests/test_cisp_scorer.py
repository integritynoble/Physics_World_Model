"""Comprehensive tests for CISP scorer — all functions."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Load cisp_scorer via importlib since 2026_CISP starts with a digit
_spec = importlib.util.spec_from_file_location(
    "cisp_scorer",
    str(Path(__file__).resolve().parent.parent / "challenges" / "2026_CISP" / "scoring" / "cisp_scorer.py"),
)
cisp_scorer = importlib.util.module_from_spec(_spec)
sys.modules["cisp_scorer"] = cisp_scorer
_spec.loader.exec_module(cisp_scorer)

# Aliases
TRACKS = cisp_scorer.TRACKS
SAFETY_BRAKES = cisp_scorer.SAFETY_BRAKES
ModalityScore = cisp_scorer.ModalityScore
TrackScore = cisp_scorer.TrackScore
CISPSubmissionScore = cisp_scorer.CISPSubmissionScore
compute_rho = cisp_scorer.compute_rho
compute_oracle_gap = cisp_scorer.compute_oracle_gap
compute_roic = cisp_scorer.compute_roic
compute_harmonic_mean = cisp_scorer.compute_harmonic_mean
compute_rho_cross = cisp_scorer.compute_rho_cross
apply_anti_goodhart = cisp_scorer.apply_anti_goodhart
check_safety_brakes = cisp_scorer.check_safety_brakes
score_track = cisp_scorer.score_track
score_submission = cisp_scorer.score_submission
load_runbundle_scores = cisp_scorer.load_runbundle_scores
export_scores = cisp_scorer.export_scores


# ── TRACKS registry ────────────────────────────────────────────────

class TestTracks:
    def test_all_tracks_present(self):
        assert set(TRACKS.keys()) == {"correct", "temporal", "medical", "cross"}

    def test_weights_sum_to_one(self):
        total = sum(t["weight"] for t in TRACKS.values())
        assert abs(total - 1.0) < 1e-10


# ── compute_rho ─────────────────────────────────────────────────────

class TestComputeRho:
    def test_basic(self):
        assert abs(compute_rho(40, 30, 37) - 0.7) < 1e-10

    def test_perfect(self):
        assert abs(compute_rho(40, 30, 40) - 1.0) < 1e-10

    def test_no_recovery(self):
        assert abs(compute_rho(40, 30, 30) - 0.0) < 1e-10

    def test_zero_denom(self):
        assert compute_rho(30, 30, 30) == 0.0

    def test_rho_above_one(self):
        assert abs(compute_rho(40, 30, 45) - 1.5) < 1e-10


# ── compute_oracle_gap ──────────────────────────────────────────────

class TestComputeOracleGap:
    def test_positive_gap(self):
        assert abs(compute_oracle_gap(40, 37) - 3.0) < 1e-10

    def test_zero_gap(self):
        assert abs(compute_oracle_gap(40, 40) - 0.0) < 1e-10


# ── compute_roic ────────────────────────────────────────────────────

class TestComputeRoic:
    def test_one_hour(self):
        assert abs(compute_roic(5.0, 3600.0) - 5.0) < 1e-10

    def test_two_hours(self):
        assert abs(compute_roic(10.0, 7200.0) - 5.0) < 1e-10

    def test_zero_compute(self):
        assert compute_roic(5.0, 0.0) == 0.0


# ── compute_harmonic_mean ───────────────────────────────────────────

class TestComputeHarmonicMean:
    def test_equal(self):
        assert abs(compute_harmonic_mean([2.0, 2.0]) - 2.0) < 1e-10

    def test_different(self):
        assert abs(compute_harmonic_mean([1.0, 3.0]) - 1.5) < 1e-10

    def test_zero_input(self):
        assert compute_harmonic_mean([0.0, 1.0]) == 0.0

    def test_negative_input(self):
        assert compute_harmonic_mean([-1.0, 1.0]) == 0.0

    def test_empty(self):
        assert compute_harmonic_mean([]) == 0.0


# ── compute_rho_cross ──────────────────────────────────────────────

class TestComputeRhoCross:
    def test_equal_rhos(self):
        assert abs(compute_rho_cross({"cassi": 0.8, "spc": 0.8}) - 0.8) < 1e-10

    def test_one_zero_kills(self):
        assert compute_rho_cross({"cassi": 0.0, "spc": 0.9}) == 0.0


# ── check_safety_brakes ────────────────────────────────────────────

class TestCheckSafetyBrakes:
    def _ms(self, rho):
        return ModalityScore("cassi", rho, 2.0, 12.0, 40, 30, 38, 37, 5)

    def test_above_threshold(self):
        assert len(check_safety_brakes(self._ms(0.85))) == 0

    def test_at_threshold(self):
        assert len(check_safety_brakes(self._ms(0.30))) == 0

    def test_below_threshold(self):
        v = check_safety_brakes(self._ms(0.29))
        assert len(v) == 1
        assert v[0]["brake"] == "recovery_ratio_regression"
        assert v[0]["action"] == "block_deployment"


# ── apply_anti_goodhart ────────────────────────────────────────────

class TestApplyAntiGoodhart:
    def test_clean(self):
        f, p = apply_anti_goodhart(0.8, 0.9, {})
        assert abs(f - (0.3 * 0.8 + 0.7 * 0.9)) < 1e-10
        assert len(p) == 0

    def test_wrong_triad(self):
        f, p = apply_anti_goodhart(0.8, 0.9, {"wrong_triad_attribution": True})
        assert "wrong_triad" in p
        assert abs(f - max(0, 0.3 * 0.8 + 0.7 * 0.9 - 0.15)) < 1e-10

    def test_overconfident_uncertainty(self):
        f, p = apply_anti_goodhart(0.8, 0.9, {"uncertainty_coverage": 0.5})
        assert "overconfident_uncertainty" in p

    def test_identifiability(self):
        f, p = apply_anti_goodhart(0.8, 0.9, {"identifiability_inconsistent": True})
        assert "identifiability" in p

    def test_dq(self):
        f, p = apply_anti_goodhart(0.8, 0.9, {"budget_ratio": 3.0})
        assert f == 0.0
        assert p["compute_dishonesty"] == "DQ"

    def test_stacked_penalties(self):
        f, p = apply_anti_goodhart(
            0.8,
            0.9,
            {
                "wrong_triad_attribution": True,
                "uncertainty_coverage": 0.5,
                "identifiability_inconsistent": True,
            },
        )
        raw = 0.3 * 0.8 + 0.7 * 0.9
        penalty = -0.15 + -0.10 + -0.10
        assert abs(f - max(0, raw + penalty)) < 1e-10

    def test_floor_at_zero(self):
        # Very low scores + heavy penalties should floor at 0
        f, _ = apply_anti_goodhart(
            0.1,
            0.1,
            {
                "wrong_triad_attribution": True,
                "uncertainty_coverage": 0.5,
                "identifiability_inconsistent": True,
            },
        )
        assert f == 0.0


# ── score_track ─────────────────────────────────────────────────────

class TestScoreTrack:
    def test_correct_track(self):
        ms1 = ModalityScore("cassi", 0.85, 2.0, 12.0, 40, 30, 38.5, 37, 5)
        ms2 = ModalityScore("cassi", 0.75, 4.0, 6.0, 40, 30, 37.5, 36, 3)
        ts = score_track("correct", [ms1, ms2])
        assert abs(ts.primary_score - 0.80) < 1e-10
        assert not ts.disqualified
        assert ts.final_score > 0
        assert "oracle_gap_mean" in ts.secondary_scores
        assert "roic_mean" in ts.secondary_scores

    def test_cross_track(self):
        ms_c = ModalityScore("cassi", 0.85, 2, 12, 40, 30, 38.5, 37, 5)
        ms_s = ModalityScore("spc", 0.70, 3, 8, 38, 28, 35, 34, 5)
        ts = score_track("cross", [ms_c, ms_s])
        hm = 2 / (1 / 0.85 + 1 / 0.70)
        assert abs(ts.primary_score - hm) < 1e-10
        assert ts.secondary_scores["n_above_050"] == 2
        assert ts.secondary_scores["n_above_080"] == 1

    def test_dq_from_diagnostics(self):
        ms = ModalityScore("cassi", 0.85, 2.0, 12.0, 40, 30, 38.5, 37, 5)
        ts = score_track("correct", [ms], diagnostics={"budget_ratio": 3.0})
        assert ts.disqualified
        assert ts.final_score == 0.0


# ── score_submission ────────────────────────────────────────────────

class TestScoreSubmission:
    def test_composite(self):
        ms = ModalityScore("cassi", 0.85, 2.0, 12.0, 40, 30, 38.5, 37, 5)
        ts = score_track("correct", [ms])
        sub = score_submission("team", {"correct": ts})
        expected = 0.35 * ts.final_score
        assert abs(sub.composite_score - expected) < 1e-10

    def test_dq_track_contributes_zero(self):
        ms = ModalityScore("cassi", 0.85, 2.0, 12.0, 40, 30, 38.5, 37, 5)
        ts_ok = score_track("correct", [ms])
        ts_dq = score_track("correct", [ms], diagnostics={"budget_ratio": 3.0})
        sub = score_submission("team", {"correct": ts_dq, "temporal": ts_ok})
        # DQ track = 0, temporal uses correct's score but temporal weight
        expected = 0.25 * ts_ok.final_score
        assert abs(sub.composite_score - expected) < 1e-10


# ── load_runbundle_scores ───────────────────────────────────────────

class TestLoadRunbundleScores:
    def test_valid(self, tmp_path):
        (tmp_path / "metrics.json").write_text(json.dumps({"psnr_db": 30.0}))
        loaded = load_runbundle_scores(tmp_path)
        assert loaded["psnr_db"] == 30.0

    def test_missing(self):
        with pytest.raises(FileNotFoundError):
            load_runbundle_scores(Path("/nonexistent"))


# ── export_scores ───────────────────────────────────────────────────

class TestExportScores:
    def test_roundtrip(self, tmp_path):
        ms = ModalityScore("cassi", 0.85, 2.0, 12.0, 40, 30, 38.5, 37, 5)
        ts = score_track("correct", [ms])
        sub = score_submission("team_test", {"correct": ts})
        out = tmp_path / "scores.json"
        export_scores(sub, out)
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert data["team_id"] == "team_test"
        assert "correct" in data["tracks"]
        assert data["tracks"]["correct"]["primary_score"] == ts.primary_score
