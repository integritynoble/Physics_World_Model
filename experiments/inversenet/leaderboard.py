"""Score and rank InverseNet submissions per task.

Reads ``baseline_results.jsonl`` (or any results file in the same format)
and produces a per-task leaderboard as a markdown table.

Usage::

    python -m experiments.inversenet.leaderboard \\
        --results results/inversenet/baseline_results.jsonl \\
        --out results/inversenet/leaderboard.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Scoring rules per task ──────────────────────────────────────────────

# T1: lower theta_error_rmse is better
# T2: higher accuracy / F1 is better
# T3: higher psnr_gain_db is better (calibration improvement)
# T4: higher psnr_db is better

TASK_PRIMARY_METRIC = {
    "T1_param_estimation": ("theta_error_rmse", "lower"),
    "T2_mismatch_id": ("accuracy", "higher"),
    "T3_calibration": ("psnr_gain_db", "higher"),
    "T4_reconstruction": ("psnr_db", "higher"),
}


@dataclass
class LeaderboardRow:
    rank: int
    baseline: str
    modality: str
    metric_mean: float
    metric_std: float
    n_samples: int


def _load_results(path: str) -> List[Dict[str, Any]]:
    """Load results JSONL."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_leaderboard(
    results: List[Dict[str, Any]],
) -> Dict[str, List[LeaderboardRow]]:
    """Build leaderboard tables keyed by task name."""

    # Group by (task, modality, baseline)
    groups: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        task = r["task"]
        modality = r["modality"]
        baseline = r["baseline"]
        metric_name, _ = TASK_PRIMARY_METRIC.get(task, ("psnr_db", "higher"))
        val = r.get("metrics", {}).get(metric_name, float("nan"))
        key = f"{modality}|{baseline}"
        groups[task][key].append(val)

    leaderboards: Dict[str, List[LeaderboardRow]] = {}

    for task, entries in groups.items():
        _, direction = TASK_PRIMARY_METRIC.get(task, ("psnr_db", "higher"))
        rows: List[LeaderboardRow] = []
        for key, vals in entries.items():
            modality, baseline = key.split("|")
            vals_clean = [v for v in vals if not np.isnan(v)]
            if not vals_clean:
                continue
            rows.append(LeaderboardRow(
                rank=0,
                baseline=baseline,
                modality=modality,
                metric_mean=float(np.mean(vals_clean)),
                metric_std=float(np.std(vals_clean)),
                n_samples=len(vals_clean),
            ))

        # Sort and assign ranks
        reverse = direction == "higher"
        rows.sort(key=lambda r: r.metric_mean, reverse=reverse)
        for i, row in enumerate(rows):
            row.rank = i + 1

        leaderboards[task] = rows

    return leaderboards


def format_markdown(
    leaderboards: Dict[str, List[LeaderboardRow]],
) -> str:
    """Format leaderboards as markdown tables."""
    lines = ["# InverseNet Leaderboard", ""]

    for task in sorted(leaderboards.keys()):
        rows = leaderboards[task]
        metric_name, direction = TASK_PRIMARY_METRIC.get(task, ("metric", "higher"))
        arrow = "^" if direction == "higher" else "v"

        lines.append(f"## {task}")
        lines.append("")
        lines.append(
            f"| Rank | Baseline | Modality | "
            f"{metric_name} (mean +/- std) {arrow} | N |"
        )
        lines.append("|------|----------|----------|" + "-" * 35 + "|---|")

        for row in rows:
            lines.append(
                f"| {row.rank} | {row.baseline} | {row.modality} | "
                f"{row.metric_mean:.4f} +/- {row.metric_std:.4f} | "
                f"{row.n_samples} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_leaderboard(
    results_path: str,
    out_path: Optional[str] = None,
) -> str:
    """Load results, build leaderboard, and optionally save to file."""
    results = _load_results(results_path)
    leaderboards = build_leaderboard(results)
    md = format_markdown(leaderboards)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(md)
        logger.info(f"Leaderboard -> {out_path}")

    return md


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate InverseNet leaderboard from results"
    )
    parser.add_argument(
        "--results",
        default="results/inversenet/baseline_results.jsonl",
    )
    parser.add_argument(
        "--out",
        default="results/inversenet/leaderboard.md",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    md = generate_leaderboard(args.results, args.out)
    print(md)


if __name__ == "__main__":
    main()
