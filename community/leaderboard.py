#!/usr/bin/env python3
"""Score RunBundle submissions and produce a leaderboard for PWM weekly challenges.

Reads submitted RunBundles from a submissions directory, scores them against
the expected.json reference metrics, and outputs a leaderboard.md file.

Usage:
    python community/leaderboard.py --week 2026-W10
    python community/leaderboard.py --week 2026-W10 --submissions-dir ./submissions
    python community/leaderboard.py --week 2026-W10 --output leaderboard.md

Scoring:
    - Primary metric (e.g., psnr_db): higher is better
    - Secondary metric (e.g., ssim): higher is better
    - Runtime: lower is better (tiebreaker)
    - Submissions below minimum thresholds are marked as INVALID
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


CHALLENGES_DIR = Path(__file__).parent / "challenges"


@dataclass
class ScoredSubmission:
    """A scored challenge submission."""

    name: str
    spec_id: str
    psnr_db: float
    ssim: float
    runtime_s: float
    theta_error: Optional[float] = None
    valid: bool = True
    invalid_reason: str = ""
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def rank_key(self) -> tuple:
        """Sort key: primary metric desc, secondary desc, runtime asc."""
        if not self.valid:
            return (0, 0.0, 0.0, 1e9)
        return (1, self.psnr_db, self.ssim, -self.runtime_s)


def load_expected(week_id: str) -> dict:
    """Load the expected.json for a given week."""
    expected_path = CHALLENGES_DIR / week_id / "expected.json"
    if not expected_path.is_file():
        raise FileNotFoundError(
            f"No expected.json found for week {week_id} at {expected_path}"
        )
    with open(expected_path) as f:
        return json.load(f)


def score_submission(
    manifest: dict,
    expected: dict,
    name: str = "unknown",
) -> ScoredSubmission:
    """Score a single submission against expected metrics.

    Args:
        manifest: The parsed runbundle_manifest.json.
        expected: The parsed expected.json for the challenge.
        name: Display name for the submission.

    Returns:
        ScoredSubmission with metrics and validity flag.
    """
    metrics = manifest.get("metrics", {})
    thresholds = expected.get("thresholds", {})

    psnr = metrics.get("psnr_db", float("nan"))
    ssim = metrics.get("ssim", float("nan"))
    runtime = metrics.get("runtime_s", float("nan"))
    theta_error = metrics.get("theta_error_rmse")

    sub = ScoredSubmission(
        name=name,
        spec_id=manifest.get("spec_id", "unknown"),
        psnr_db=psnr,
        ssim=ssim,
        runtime_s=runtime,
        theta_error=theta_error,
    )

    # Check finite values
    for metric_name, value in [("psnr_db", psnr), ("ssim", ssim), ("runtime_s", runtime)]:
        if not math.isfinite(value):
            sub.valid = False
            sub.invalid_reason = f"{metric_name} is not finite"
            return sub

    # Check thresholds
    psnr_min = thresholds.get("psnr_db_min", 0)
    if psnr < psnr_min:
        sub.valid = False
        sub.invalid_reason = f"PSNR {psnr:.2f} dB below minimum {psnr_min:.2f} dB"
        return sub

    ssim_min = thresholds.get("ssim_min", 0)
    if ssim < ssim_min:
        sub.valid = False
        sub.invalid_reason = f"SSIM {ssim:.4f} below minimum {ssim_min:.4f}"
        return sub

    runtime_max = thresholds.get("runtime_s_max", float("inf"))
    if runtime > runtime_max:
        sub.valid = False
        sub.invalid_reason = f"Runtime {runtime:.1f}s exceeds maximum {runtime_max:.1f}s"
        return sub

    # Collect extra metrics
    for key, value in metrics.items():
        if key not in ("psnr_db", "ssim", "runtime_s") and isinstance(value, (int, float)):
            sub.extra_metrics[key] = value

    return sub


def collect_submissions(
    submissions_dir: Path,
) -> List[dict]:
    """Collect all RunBundle manifests from a submissions directory.

    Each submission is either:
    - A directory containing runbundle_manifest.json
    - A .json file that is a manifest itself

    Returns list of (name, manifest_dict) tuples.
    """
    results = []
    if not submissions_dir.is_dir():
        return results

    for entry in sorted(submissions_dir.iterdir()):
        if entry.is_dir():
            manifest_path = entry / "runbundle_manifest.json"
            if manifest_path.is_file():
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    results.append((entry.name, manifest))
                except (json.JSONDecodeError, OSError):
                    pass
        elif entry.suffix == ".json" and entry.name != "expected.json":
            try:
                with open(entry) as f:
                    manifest = json.load(f)
                if "version" in manifest and "metrics" in manifest:
                    results.append((entry.stem, manifest))
            except (json.JSONDecodeError, OSError):
                pass

    return results


def generate_leaderboard_md(
    week_id: str,
    expected: dict,
    submissions: List[ScoredSubmission],
) -> str:
    """Generate a Markdown leaderboard table.

    Args:
        week_id: The challenge week identifier.
        expected: The expected.json content.
        submissions: List of scored submissions, will be sorted.

    Returns:
        Markdown string with leaderboard table.
    """
    modality = expected.get("modality", "unknown")
    ref = expected.get("reference_metrics", {})
    primary = expected.get("primary_metric", "psnr_db")
    secondary = expected.get("secondary_metric", "ssim")

    # Sort: valid entries first, then by rank key descending
    submissions.sort(key=lambda s: s.rank_key, reverse=True)

    lines = [
        f"# PWM Challenge Leaderboard: {week_id}",
        "",
        f"**Modality:** {modality}  ",
        f"**Primary metric:** {primary}  ",
        f"**Secondary metric:** {secondary}  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "## Reference Baseline",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| PSNR (dB) | {ref.get('psnr_db', 'N/A')} |",
        f"| SSIM | {ref.get('ssim', 'N/A')} |",
        f"| Runtime (s) | {ref.get('runtime_s', 'N/A')} |",
        "",
    ]

    if not submissions:
        lines.append("## Submissions")
        lines.append("")
        lines.append("_No submissions received yet._")
        lines.append("")
    else:
        lines.append("## Rankings")
        lines.append("")

        has_theta = any(s.theta_error is not None for s in submissions)
        header = "| Rank | Name | PSNR (dB) | SSIM | Runtime (s) |"
        sep = "|------|------|-----------|------|-------------|"
        if has_theta:
            header += " theta-err |"
            sep += "-----------|"
        header += " Status |"
        sep += "--------|"

        lines.append(header)
        lines.append(sep)

        rank = 0
        for sub in submissions:
            if sub.valid:
                rank += 1
                rank_str = str(rank)
                status = "OK"
            else:
                rank_str = "-"
                status = f"INVALID: {sub.invalid_reason}"

            row = (
                f"| {rank_str} "
                f"| {sub.name} "
                f"| {sub.psnr_db:.2f} "
                f"| {sub.ssim:.4f} "
                f"| {sub.runtime_s:.1f} |"
            )
            if has_theta:
                te = f"{sub.theta_error:.4f}" if sub.theta_error is not None else "N/A"
                row += f" {te} |"
            row += f" {status} |"
            lines.append(row)

        lines.append("")

    lines.append("---")
    lines.append(f"*{len(submissions)} submission(s) evaluated.*")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PWM challenge leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--week",
        type=str,
        required=True,
        help="Challenge week ID (e.g., 2026-W10)",
    )
    parser.add_argument(
        "--submissions-dir",
        type=str,
        default=None,
        help=(
            "Directory containing submission RunBundles. "
            "Defaults to community/challenges/<week>/submissions/"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for leaderboard.md. Defaults to challenge dir.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed progress"
    )
    args = parser.parse_args()

    week_id = args.week
    challenge_dir = CHALLENGES_DIR / week_id

    if not challenge_dir.is_dir():
        print(f"ERROR: Challenge directory not found: {challenge_dir}")
        sys.exit(1)

    # Load expected metrics
    try:
        expected = load_expected(week_id)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if args.verbose:
        print(f"Challenge: {week_id}")
        print(f"Modality: {expected.get('modality', 'unknown')}")
        ref = expected.get("reference_metrics", {})
        print(f"Reference PSNR: {ref.get('psnr_db', 'N/A')} dB")

    # Collect submissions
    submissions_dir = Path(args.submissions_dir) if args.submissions_dir else (
        challenge_dir / "submissions"
    )

    raw_submissions = collect_submissions(submissions_dir)
    if args.verbose:
        print(f"Found {len(raw_submissions)} submission(s) in {submissions_dir}")

    # Score each submission
    scored = []
    for name, manifest in raw_submissions:
        sub = score_submission(manifest, expected, name=name)
        scored.append(sub)
        if args.verbose:
            status = "VALID" if sub.valid else f"INVALID ({sub.invalid_reason})"
            print(f"  {name}: PSNR={sub.psnr_db:.2f} dB, SSIM={sub.ssim:.4f}, {status}")

    # Generate leaderboard
    md = generate_leaderboard_md(week_id, expected, scored)

    # Write output
    output_path = Path(args.output) if args.output else (challenge_dir / "leaderboard.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md)

    print(f"Leaderboard written to {output_path}")
    if scored:
        valid_count = sum(1 for s in scored if s.valid)
        print(f"  {valid_count}/{len(scored)} valid submissions")
    else:
        print("  No submissions found (empty leaderboard generated)")


if __name__ == "__main__":
    main()
