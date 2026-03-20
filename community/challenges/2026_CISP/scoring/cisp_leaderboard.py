"""
CISP 2026 Leaderboard
=====================

Extends community/leaderboard.py with CISP-specific ranking logic.

Features:
- Per-track ranking
- Cross-track composite ranking
- Anti-Goodhart enforcement
- Baseline comparison
- Live updates during submission window
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from .cisp_scorer import (
        TRACKS,
        CISPSubmissionScore,
        score_submission,
    )
except ImportError:
    from cisp_scorer import (
        TRACKS,
        CISPSubmissionScore,
        score_submission,
    )


# ---------------------------------------------------------------------------
# Leaderboard entry
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardEntry:
    """A single entry on the CISP leaderboard."""
    team_id: str
    track: str
    primary_score: float
    secondary_scores: dict
    final_score: float
    rank: int
    submission_time: str
    is_baseline: bool = False
    is_disqualified: bool = False
    badge: str = ""  # "reproducible", "identified", "anonymous"


@dataclass
class CISPLeaderboard:
    """Complete CISP leaderboard state."""
    track_boards: dict = field(default_factory=dict)  # track -> List[LeaderboardEntry]
    composite_board: list = field(default_factory=list)
    last_updated: str = ""
    version: str = "2026.1"


# ---------------------------------------------------------------------------
# Leaderboard operations
# ---------------------------------------------------------------------------


def create_leaderboard() -> CISPLeaderboard:
    """Create an empty CISP leaderboard."""
    board = CISPLeaderboard(
        track_boards={track: [] for track in TRACKS},
        composite_board=[],
        last_updated=datetime.now(timezone.utc).isoformat(),
    )
    return board


def add_submission(board: CISPLeaderboard,
                   submission: CISPSubmissionScore,
                   badge: str = "anonymous") -> CISPLeaderboard:
    """Add a scored submission to the leaderboard and re-rank."""
    now = datetime.now(timezone.utc).isoformat()

    for track_name, ts in submission.track_scores.items():
        entry = LeaderboardEntry(
            team_id=submission.team_id,
            track=track_name,
            primary_score=ts.primary_score,
            secondary_scores=ts.secondary_scores,
            final_score=ts.final_score,
            rank=0,  # will be set by re-ranking
            submission_time=now,
            is_disqualified=ts.disqualified,
            badge=badge,
        )
        if track_name not in board.track_boards:
            board.track_boards[track_name] = []
        board.track_boards[track_name].append(entry)

    # Composite entry
    composite_entry = LeaderboardEntry(
        team_id=submission.team_id,
        track="composite",
        primary_score=submission.composite_score,
        secondary_scores={},
        final_score=submission.composite_score,
        rank=0,
        submission_time=now,
        badge=badge,
    )
    board.composite_board.append(composite_entry)

    # Re-rank all tracks
    board = _rerank(board)
    board.last_updated = now
    return board


def add_baseline(board: CISPLeaderboard, baseline_name: str,
                 track: str, score: float) -> CISPLeaderboard:
    """Add a frozen baseline entry to a track."""
    entry = LeaderboardEntry(
        team_id=f"BASELINE:{baseline_name}",
        track=track,
        primary_score=score,
        secondary_scores={},
        final_score=score,
        rank=0,
        submission_time="2026-01-01T00:00:00",
        is_baseline=True,
        badge="baseline",
    )
    if track not in board.track_boards:
        board.track_boards[track] = []
    board.track_boards[track].append(entry)
    board = _rerank(board)
    return board


def _rerank(board: CISPLeaderboard) -> CISPLeaderboard:
    """Re-rank all entries on all track boards and composite."""
    for track_name, entries in board.track_boards.items():
        # Sort by final_score descending (DQ entries at bottom)
        active = [e for e in entries if not e.is_disqualified]
        dq = [e for e in entries if e.is_disqualified]

        active.sort(key=lambda e: e.final_score, reverse=True)
        for i, entry in enumerate(active):
            entry.rank = i + 1
        for entry in dq:
            entry.rank = -1  # DQ

        board.track_boards[track_name] = active + dq

    # Composite
    active = [e for e in board.composite_board if not e.is_disqualified]
    active.sort(key=lambda e: e.final_score, reverse=True)
    for i, entry in enumerate(active):
        entry.rank = i + 1
    board.composite_board = active

    return board


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def format_track_board(board: CISPLeaderboard, track: str,
                       top_n: int = 20) -> str:
    """Format a track leaderboard as a text table."""
    entries = board.track_boards.get(track, [])[:top_n]
    if not entries:
        return f"No submissions for track: {track}"

    lines = [
        f"CISP 2026 - Track: {track.upper()}",
        f"{'Rank':>4}  {'Team':<30}  {'Score':>8}  {'Badge':<14}  {'Status':<6}",
        "-" * 72,
    ]
    for e in entries:
        status = "DQ" if e.is_disqualified else "OK"
        rank_str = str(e.rank) if e.rank > 0 else "DQ"
        lines.append(
            f"{rank_str:>4}  {e.team_id:<30}  {e.final_score:>8.4f}  "
            f"{e.badge:<14}  {status:<6}"
        )
    return "\n".join(lines)


def format_composite_board(board: CISPLeaderboard, top_n: int = 20) -> str:
    """Format the composite leaderboard."""
    entries = board.composite_board[:top_n]
    if not entries:
        return "No composite submissions yet."

    lines = [
        "CISP 2026 - COMPOSITE RANKING",
        f"{'Rank':>4}  {'Team':<30}  {'Score':>8}  {'Badge':<14}",
        "-" * 64,
    ]
    for e in entries:
        lines.append(
            f"{e.rank:>4}  {e.team_id:<30}  {e.final_score:>8.4f}  {e.badge:<14}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_leaderboard(board: CISPLeaderboard, path: Path):
    """Save leaderboard state to JSON."""
    data = {
        "version": board.version,
        "last_updated": board.last_updated,
        "track_boards": {},
        "composite_board": [],
    }

    for track_name, entries in board.track_boards.items():
        data["track_boards"][track_name] = [
            {
                "team_id": e.team_id,
                "track": e.track,
                "primary_score": e.primary_score,
                "secondary_scores": e.secondary_scores,
                "final_score": e.final_score,
                "rank": e.rank,
                "submission_time": e.submission_time,
                "is_baseline": e.is_baseline,
                "is_disqualified": e.is_disqualified,
                "badge": e.badge,
            }
            for e in entries
        ]

    for e in board.composite_board:
        data["composite_board"].append({
            "team_id": e.team_id,
            "final_score": e.final_score,
            "rank": e.rank,
            "submission_time": e.submission_time,
            "badge": e.badge,
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_leaderboard(path: Path) -> CISPLeaderboard:
    """Load leaderboard state from JSON."""
    with open(path) as f:
        data = json.load(f)

    board = CISPLeaderboard(
        version=data.get("version", "2026.1"),
        last_updated=data.get("last_updated", ""),
    )

    for track_name, entries_data in data.get("track_boards", {}).items():
        board.track_boards[track_name] = [
            LeaderboardEntry(**e) for e in entries_data
        ]

    board.composite_board = [
        LeaderboardEntry(
            team_id=e["team_id"],
            track="composite",
            primary_score=e["final_score"],
            secondary_scores={},
            final_score=e["final_score"],
            rank=e["rank"],
            submission_time=e["submission_time"],
            badge=e.get("badge", "anonymous"),
        )
        for e in data.get("composite_board", [])
    ]

    return board
