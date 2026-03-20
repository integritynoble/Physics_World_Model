"""Comprehensive tests for CISP leaderboard — all functions."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load cisp_scorer first
_scorer_path = Path(__file__).resolve().parent.parent / "challenges" / "2026_CISP" / "scoring" / "cisp_scorer.py"
_spec1 = importlib.util.spec_from_file_location("cisp_scorer", str(_scorer_path))
cisp_scorer = importlib.util.module_from_spec(_spec1)
sys.modules["cisp_scorer"] = cisp_scorer
_spec1.loader.exec_module(cisp_scorer)

# Load cisp_leaderboard
_lb_path = Path(__file__).resolve().parent.parent / "challenges" / "2026_CISP" / "scoring" / "cisp_leaderboard.py"
_spec2 = importlib.util.spec_from_file_location("cisp_leaderboard", str(_lb_path))
cisp_leaderboard = importlib.util.module_from_spec(_spec2)
sys.modules["cisp_leaderboard"] = cisp_leaderboard
_spec2.loader.exec_module(cisp_leaderboard)

# Aliases
ModalityScore = cisp_scorer.ModalityScore
score_track = cisp_scorer.score_track
score_submission_cisp = cisp_scorer.score_submission

create_leaderboard = cisp_leaderboard.create_leaderboard
add_submission = cisp_leaderboard.add_submission
add_baseline = cisp_leaderboard.add_baseline
format_track_board = cisp_leaderboard.format_track_board
format_composite_board = cisp_leaderboard.format_composite_board
save_leaderboard = cisp_leaderboard.save_leaderboard
load_leaderboard = cisp_leaderboard.load_leaderboard
CISPLeaderboard = cisp_leaderboard.CISPLeaderboard
LeaderboardEntry = cisp_leaderboard.LeaderboardEntry


def _make_submission(team_id, rho=0.85):
    ms = ModalityScore("cassi", rho, 2.0, 12.0, 40, 30, 38.5, 37, 5)
    ts = score_track("correct", [ms])
    return score_submission_cisp(team_id, {"correct": ts})


# ── create_leaderboard ──────────────────────────────────────────────

class TestCreateLeaderboard:
    def test_all_tracks(self):
        board = create_leaderboard()
        assert set(board.track_boards.keys()) == {"correct", "temporal", "medical", "cross"}
        for entries in board.track_boards.values():
            assert entries == []

    def test_empty_composite(self):
        board = create_leaderboard()
        assert board.composite_board == []

    def test_has_timestamp(self):
        board = create_leaderboard()
        assert board.last_updated != ""


# ── add_submission ──────────────────────────────────────────────────

class TestAddSubmission:
    def test_single(self):
        board = create_leaderboard()
        sub = _make_submission("alpha")
        board = add_submission(board, sub, badge="reproducible")
        assert len(board.track_boards["correct"]) == 1
        assert board.track_boards["correct"][0].team_id == "alpha"
        assert board.track_boards["correct"][0].badge == "reproducible"
        assert len(board.composite_board) == 1

    def test_ranking_two_teams(self):
        board = create_leaderboard()
        board = add_submission(board, _make_submission("alpha", rho=0.90))
        board = add_submission(board, _make_submission("beta", rho=0.70))
        entries = board.track_boards["correct"]
        assert entries[0].team_id == "alpha"
        assert entries[0].rank == 1
        assert entries[1].team_id == "beta"
        assert entries[1].rank == 2

    def test_composite_ranked(self):
        board = create_leaderboard()
        board = add_submission(board, _make_submission("alpha", rho=0.90))
        board = add_submission(board, _make_submission("beta", rho=0.70))
        assert board.composite_board[0].team_id == "alpha"
        assert board.composite_board[0].rank == 1


# ── add_baseline ────────────────────────────────────────────────────

class TestAddBaseline:
    def test_baseline_added(self):
        board = create_leaderboard()
        board = add_baseline(board, "TwIST", "correct", 0.50)
        entries = board.track_boards["correct"]
        assert any(e.is_baseline for e in entries)
        assert any("BASELINE" in e.team_id for e in entries)

    def test_baseline_ranked_with_submissions(self):
        board = create_leaderboard()
        board = add_submission(board, _make_submission("alpha", rho=0.90))
        board = add_baseline(board, "TwIST", "correct", 0.50)
        entries = board.track_boards["correct"]
        # alpha should rank above baseline
        alpha = next(e for e in entries if e.team_id == "alpha")
        baseline = next(e for e in entries if e.is_baseline)
        assert alpha.rank < baseline.rank


# ── format_track_board ──────────────────────────────────────────────

class TestFormatTrackBoard:
    def test_empty(self):
        board = create_leaderboard()
        txt = format_track_board(board, "correct")
        assert "No submissions" in txt

    def test_with_entries(self):
        board = create_leaderboard()
        board = add_submission(board, _make_submission("alpha"))
        txt = format_track_board(board, "correct")
        assert "alpha" in txt
        assert "CORRECT" in txt

    def test_top_n(self):
        board = create_leaderboard()
        for i in range(5):
            board = add_submission(board, _make_submission(f"team_{i}", rho=0.80 - i * 0.05))
        txt = format_track_board(board, "correct", top_n=3)
        assert "team_0" in txt
        assert "team_4" not in txt


# ── format_composite_board ──────────────────────────────────────────

class TestFormatCompositeBoard:
    def test_empty(self):
        board = create_leaderboard()
        txt = format_composite_board(board)
        assert "No composite" in txt

    def test_with_entries(self):
        board = create_leaderboard()
        board = add_submission(board, _make_submission("alpha"))
        txt = format_composite_board(board)
        assert "alpha" in txt
        assert "COMPOSITE" in txt


# ── save_leaderboard / load_leaderboard ─────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        board = create_leaderboard()
        board = add_submission(board, _make_submission("alpha", rho=0.90))
        board = add_submission(board, _make_submission("beta", rho=0.70))
        board = add_baseline(board, "TwIST", "correct", 0.50)

        path = tmp_path / "board.json"
        save_leaderboard(board, path)
        assert path.exists()

        loaded = load_leaderboard(path)
        assert loaded.track_boards["correct"][0].team_id == "alpha"
        assert loaded.track_boards["correct"][0].rank == 1
        assert len(loaded.composite_board) == 2

    def test_roundtrip_preserves_baseline(self, tmp_path):
        board = create_leaderboard()
        board = add_baseline(board, "TwIST", "correct", 0.50)
        path = tmp_path / "board.json"
        save_leaderboard(board, path)
        loaded = load_leaderboard(path)
        assert any(e.is_baseline for e in loaded.track_boards["correct"])

    def test_json_valid(self, tmp_path):
        board = create_leaderboard()
        board = add_submission(board, _make_submission("alpha"))
        path = tmp_path / "board.json"
        save_leaderboard(board, path)
        with open(path) as f:
            data = json.load(f)
        assert "track_boards" in data
        assert "composite_board" in data
        assert data["version"] == "2026.1"
