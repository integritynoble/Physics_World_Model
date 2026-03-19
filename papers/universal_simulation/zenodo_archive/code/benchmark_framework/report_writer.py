"""ReportWriter – RunBundle creation and JSON/markdown report generation.

Wraps the existing ``pwm_core.core.runbundle`` infrastructure with
benchmark-specific metadata and source attribution.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from benchmarks.framework.source_attribution import SourceAttribution


# ---------------------------------------------------------------------------
# RunBundle – lightweight result container
# ---------------------------------------------------------------------------

@dataclass
class RunBundle:
    """Container for a single benchmark run's results.

    Can be serialised to JSON and optionally saved as a RunBundle directory
    using the existing ``pwm_core.core.runbundle`` writer.
    """
    modality_id: str
    level: str  # M0, M1, M2, M3, M4
    solver: str = ""
    tier: str = "C"

    # Core metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    per_algorithm: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Reference values
    reference_psnr: Optional[float] = None
    expected_psnr_range: Optional[List[float]] = None

    # Mismatch results
    mismatch_results: List[Dict[str, Any]] = field(default_factory=list)
    grid_search_result: Optional[Dict[str, Any]] = None
    rho: Optional[float] = None

    # Source attribution
    source_attribution: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: str = ""
    run_id: str = ""
    wall_time_s: float = 0.0
    config_path: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "modality_id": self.modality_id,
            "level": self.level,
            "solver": self.solver,
            "tier": self.tier,
            "metrics": self.metrics,
            "per_algorithm": self.per_algorithm,
            "reference_psnr": self.reference_psnr,
            "expected_psnr_range": self.expected_psnr_range,
            "mismatch_results": self.mismatch_results,
            "grid_search_result": self.grid_search_result,
            "rho": self.rho,
            "source_attribution": self.source_attribution,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "wall_time_s": self.wall_time_s,
            "config_path": self.config_path,
        }
        return d


# ---------------------------------------------------------------------------
# ReportWriter
# ---------------------------------------------------------------------------

class ReportWriter:
    """Write benchmark results to disk in JSON and markdown."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(__file__).parent.parent / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    def save_json(self, bundle: RunBundle) -> Path:
        """Save RunBundle as a JSON file.

        Returns:
            Path to the saved JSON.
        """
        run_dir = self.output_dir / bundle.modality_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{bundle.level}_{bundle.run_id}.json"
        with open(path, "w") as f:
            json.dump(bundle.to_dict(), f, indent=2, default=_json_default)
        return path

    # ------------------------------------------------------------------
    # Markdown summary
    # ------------------------------------------------------------------

    def save_markdown(self, bundle: RunBundle) -> Path:
        """Save a human-readable markdown summary.

        Returns:
            Path to the saved markdown file.
        """
        run_dir = self.output_dir / bundle.modality_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{bundle.level}_{bundle.run_id}.md"

        lines = [
            f"# {bundle.modality_id} – {bundle.level} Benchmark Results",
            "",
            f"**Run ID**: {bundle.run_id}  ",
            f"**Timestamp**: {bundle.timestamp}  ",
            f"**Tier**: {bundle.tier}  ",
            f"**Wall time**: {bundle.wall_time_s:.1f}s  ",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in bundle.metrics.items():
            lines.append(f"| {k} | {v:.4f} |")

        if bundle.reference_psnr is not None:
            lines.append(f"| reference_psnr | {bundle.reference_psnr:.1f} |")

        if bundle.per_algorithm:
            lines.extend(["", "## Per-Algorithm Results", ""])
            lines.append("| Algorithm | Tier | PSNR | SSIM | Params |")
            lines.append("|-----------|------|------|------|--------|")
            for algo, info in bundle.per_algorithm.items():
                psnr = info.get("psnr", "-")
                ssim = info.get("ssim", "-")
                if isinstance(psnr, float):
                    psnr = f"{psnr:.2f}"
                if isinstance(ssim, float):
                    ssim = f"{ssim:.4f}"
                lines.append(
                    f"| {algo} | {info.get('tier', '-')} | {psnr} | {ssim} | {info.get('params', '-')} |"
                )

        if bundle.mismatch_results:
            lines.extend(["", "## Mismatch Scenarios", ""])
            lines.append("| Scenario | PSNR | PSNR Drop |")
            lines.append("|----------|------|-----------|")
            for mr in bundle.mismatch_results:
                psnr = mr.get("psnr", "-")
                drop = mr.get("psnr_drop", 0)
                name = mr.get("name", "?")
                if isinstance(psnr, float):
                    psnr = f"{psnr:.2f}"
                lines.append(f"| {name} | {psnr} | {drop:.2f} dB |")

        if bundle.rho is not None:
            lines.extend(["", f"## Correction: rho = {bundle.rho:.4f}", ""])

        if bundle.source_attribution:
            lines.extend(["", "## Source Attribution", ""])
            for key, val in bundle.source_attribution.items():
                if isinstance(val, dict):
                    ref = val.get("reference", "")
                    stype = val.get("type", "")
                    lines.append(f"- **{key}**: {stype} – {ref}")
                else:
                    lines.append(f"- **{key}**: {val}")

        lines.append("")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    # ------------------------------------------------------------------
    # RunBundle directory (optional – integrates with pwm_core)
    # ------------------------------------------------------------------

    def save_runbundle(self, bundle: RunBundle, arrays: Dict[str, np.ndarray] = None) -> Path:
        """Save full RunBundle directory with artifacts.

        Attempts to use ``pwm_core.core.runbundle.writer`` if available,
        otherwise falls back to a simple directory structure.

        Args:
            bundle: The run results.
            arrays: Optional dict of arrays to save (e.g. ``{"x_hat": arr}``).

        Returns:
            Path to the RunBundle directory.
        """
        rb_dir = self.output_dir / bundle.modality_id / f"run_{bundle.run_id}"
        artifacts_dir = rb_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        with open(rb_dir / "results.json", "w") as f:
            json.dump(bundle.to_dict(), f, indent=2, default=_json_default)

        # Save arrays
        if arrays:
            for name, arr in arrays.items():
                np.save(artifacts_dir / f"{name}.npy", arr)

        return rb_dir

    # ------------------------------------------------------------------
    # Aggregate report
    # ------------------------------------------------------------------

    def save_aggregate(self, bundles: List[RunBundle]) -> Path:
        """Save an aggregate report across multiple modalities.

        Returns:
            Path to aggregate JSON.
        """
        path = self.output_dir / "aggregate_results.json"
        data = {
            "n_modalities": len(bundles),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": {},
            "results": [],
        }
        for b in bundles:
            data["results"].append(b.to_dict())
            data["summary"][b.modality_id] = {
                "level": b.level,
                "psnr": b.metrics.get("psnr"),
                "ssim": b.metrics.get("ssim"),
                "tier": b.tier,
                "rho": b.rho,
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        return path


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
