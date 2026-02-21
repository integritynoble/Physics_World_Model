"""Comprehensive tests for community/leaderboard.py — all functions."""
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from community.leaderboard import (
    ScoredSubmission,
    collect_submissions,
    generate_leaderboard_md,
    load_expected,
    score_submission,
)


# ── ScoredSubmission.rank_key ───────────────────────────────────────

class TestScoredSubmissionRankKey:
    def test_valid_beats_invalid(self):
        valid = ScoredSubmission("a", "x", 30.0, 0.9, 1.0)
        invalid = ScoredSubmission("b", "x", 40.0, 0.99, 0.1, valid=False)
        assert valid.rank_key > invalid.rank_key

    def test_higher_psnr_wins(self):
        high = ScoredSubmission("h", "x", 35.0, 0.9, 1.0)
        low = ScoredSubmission("l", "x", 25.0, 0.9, 1.0)
        assert high.rank_key > low.rank_key

    def test_higher_ssim_wins_on_psnr_tie(self):
        high = ScoredSubmission("h", "x", 30.0, 0.95, 1.0)
        low = ScoredSubmission("l", "x", 30.0, 0.80, 1.0)
        assert high.rank_key > low.rank_key

    def test_faster_wins_on_metric_tie(self):
        fast = ScoredSubmission("f", "x", 30.0, 0.9, 0.5)
        slow = ScoredSubmission("s", "x", 30.0, 0.9, 10.0)
        assert fast.rank_key > slow.rank_key


# ── load_expected ───────────────────────────────────────────────────

class TestLoadExpected:
    def test_w10(self):
        e = load_expected("2026-W10")
        assert e["modality"] == "cassi"
        assert e["primary_metric"] == "psnr_db"

    def test_w11(self):
        assert load_expected("2026-W11")["modality"] == "spc"

    def test_w12(self):
        assert load_expected("2026-W12")["modality"] == "ct"

    def test_w13(self):
        assert load_expected("2026-W13")["modality"] == "widefield"

    def test_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_expected("2099-W99")


# ── score_submission ────────────────────────────────────────────────

_THRESHOLDS = {"thresholds": {"psnr_db_min": 20.0, "ssim_min": 0.5, "runtime_s_max": 300.0}}


class TestScoreSubmission:
    def test_valid(self):
        m = {"spec_id": "t", "metrics": {"psnr_db": 32.5, "ssim": 0.95, "runtime_s": 1.2}}
        s = score_submission(m, _THRESHOLDS, name="ok")
        assert s.valid
        assert s.psnr_db == 32.5

    def test_below_psnr(self):
        m = {"spec_id": "t", "metrics": {"psnr_db": 15.0, "ssim": 0.95, "runtime_s": 1.0}}
        s = score_submission(m, _THRESHOLDS, name="low")
        assert not s.valid
        assert "PSNR" in s.invalid_reason

    def test_below_ssim(self):
        m = {"spec_id": "t", "metrics": {"psnr_db": 25.0, "ssim": 0.1, "runtime_s": 1.0}}
        s = score_submission(m, _THRESHOLDS, name="low")
        assert not s.valid
        assert "SSIM" in s.invalid_reason

    def test_over_runtime(self):
        m = {"spec_id": "t", "metrics": {"psnr_db": 25.0, "ssim": 0.9, "runtime_s": 999.0}}
        s = score_submission(m, _THRESHOLDS, name="slow")
        assert not s.valid
        assert "Runtime" in s.invalid_reason

    def test_nan_metric(self):
        m = {"spec_id": "t", "metrics": {"psnr_db": float("nan"), "ssim": 0.9, "runtime_s": 1.0}}
        s = score_submission(m, _THRESHOLDS, name="nan")
        assert not s.valid
        assert "not finite" in s.invalid_reason

    def test_missing_metrics(self):
        m = {"spec_id": "t", "metrics": {}}
        s = score_submission(m, _THRESHOLDS, name="missing")
        assert not s.valid

    def test_extra_metrics_collected(self):
        m = {
            "spec_id": "t",
            "metrics": {
                "psnr_db": 25.0,
                "ssim": 0.9,
                "runtime_s": 1.0,
                "theta_error_rmse": 0.05,
                "custom": 3.14,
            },
        }
        s = score_submission(m, _THRESHOLDS, name="extra")
        assert s.valid
        assert s.theta_error == 0.05
        assert "custom" in s.extra_metrics


# ── collect_submissions ─────────────────────────────────────────────

class TestCollectSubmissions:
    def test_dir_and_json(self, tmp_path):
        # Directory-based
        sub_dir = tmp_path / "team1"
        sub_dir.mkdir()
        with open(sub_dir / "runbundle_manifest.json", "w") as f:
            json.dump({"version": "0.3.0", "metrics": {"psnr_db": 30.0}}, f)

        # JSON-based
        with open(tmp_path / "team2.json", "w") as f:
            json.dump({"version": "0.3.0", "metrics": {"psnr_db": 28.0}}, f)

        # expected.json should be skipped
        with open(tmp_path / "expected.json", "w") as f:
            json.dump({"version": "0.3.0", "metrics": {}}, f)

        results = collect_submissions(tmp_path)
        names = [n for n, _ in results]
        assert "team1" in names
        assert "team2" in names
        assert "expected" not in names

    def test_nonexistent(self):
        assert collect_submissions(Path("/nonexistent")) == []

    def test_bad_json_skipped(self, tmp_path):
        (tmp_path / "bad.json").write_text("{invalid json")
        assert collect_submissions(tmp_path) == []


# ── generate_leaderboard_md ─────────────────────────────────────────

_EXPECTED_FULL = {
    "modality": "CASSI",
    "primary_metric": "psnr_db",
    "secondary_metric": "ssim",
    "reference_metrics": {"psnr_db": 30.0, "ssim": 0.90, "runtime_s": 2.0},
}


class TestGenerateLeaderboardMd:
    def test_with_submissions(self):
        subs = [
            ScoredSubmission("alpha", "cassi", 35.0, 0.97, 0.5),
            ScoredSubmission("beta", "cassi", 28.0, 0.85, 2.0),
            ScoredSubmission("gamma", "cassi", 0.0, 0.0, 0.0, valid=False, invalid_reason="bad"),
        ]
        md = generate_leaderboard_md("2026-W10", _EXPECTED_FULL, subs)
        assert "# PWM Challenge Leaderboard: 2026-W10" in md
        assert "CASSI" in md
        assert "alpha" in md
        assert "INVALID" in md
        assert "3 submission(s)" in md

    def test_empty(self):
        md = generate_leaderboard_md("2026-W10", _EXPECTED_FULL, [])
        assert "No submissions" in md
        assert "0 submission(s)" in md

    def test_ranking_order(self):
        subs = [
            ScoredSubmission("low", "cassi", 25.0, 0.80, 5.0),
            ScoredSubmission("high", "cassi", 35.0, 0.97, 0.5),
        ]
        md = generate_leaderboard_md("2026-W10", _EXPECTED_FULL, subs)
        lines = md.split("\n")
        high_line = next(i for i, l in enumerate(lines) if "high" in l)
        low_line = next(i for i, l in enumerate(lines) if "low" in l)
        assert high_line < low_line  # higher ranked appears first

    def test_theta_error_column(self):
        subs = [
            ScoredSubmission("a", "cassi", 35.0, 0.97, 0.5, theta_error=0.03),
            ScoredSubmission("b", "cassi", 28.0, 0.85, 2.0, theta_error=None),
        ]
        md = generate_leaderboard_md("2026-W10", _EXPECTED_FULL, subs)
        assert "theta-err" in md
        assert "0.0300" in md
        assert "N/A" in md
