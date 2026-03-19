#!/usr/bin/env python3
"""
Collect best reconstruction results for all 64 modalities.

Scans every run directory, picks the best run per modality (highest w1_psnr),
then copies the images and metrics into the platform static directory so the
web UI can display them.

Output structure:
    platform/pwm_platform/static/img/results/<modality_key>/
        x_true.png
        y.png
        x_hat.png
        comparison.png
    platform/pwm_platform/static/img/results/results_summary.json

Usage:
    python scripts/collect_recon_results.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
OUT_DIR = ROOT / "platform" / "pwm_platform" / "static" / "img" / "results"
SUMMARY_PATH = OUT_DIR / "results_summary.json"

# Canonical 64 modality keys (used for filtering out variants like spc_set11)
sys.path.insert(0, str(ROOT / "platform"))
from pwm_platform.services.modality_database import MODALITY_DATABASE  # noqa: E402

VALID_KEYS = set(MODALITY_DATABASE.keys())

# Regex to extract modality key from run directory name
# Pattern: run_<modality>_exp_<hash> or run_<modality>_benchmark_<hash>
RUN_DIR_RE = re.compile(r"^run_(.+?)_(exp|benchmark)_[0-9a-f]+$")


def _parse_run_dir(run_dir: Path) -> tuple[str, dict] | None:
    """Parse a run directory, returning (modality_key, info_dict) or None."""
    m = RUN_DIR_RE.match(run_dir.name)
    if not m:
        return None
    key = m.group(1)
    if key not in VALID_KEYS:
        return None

    images_dir = run_dir / "artifacts" / "images"
    metrics_path = run_dir / "artifacts" / "metrics.json"

    required_images = ["x_true.png", "y.png", "x_hat.png", "comparison.png"]
    if not all((images_dir / img).exists() for img in required_images):
        return None
    if not metrics_path.exists():
        return None

    try:
        metrics = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    psnr = metrics.get("w1_psnr", 0.0)
    ssim = metrics.get("w1_ssim", 0.0)
    nrmse = metrics.get("w1_nrmse", 999.0)

    # Try to determine solver from run metadata or config
    solver = "unknown"
    config_path = run_dir / "artifacts" / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            solver = cfg.get("solver", cfg.get("solver_name", "unknown"))
        except (json.JSONDecodeError, OSError):
            pass
    if solver == "unknown":
        # Fall back to the modality's default solver
        solver = MODALITY_DATABASE[key].get("default_solver", "unknown")

    return key, {
        "run_dir": str(run_dir),
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "nrmse": round(nrmse, 4),
        "solver": solver,
        "images_dir": str(images_dir),
    }


def collect_best_runs() -> dict[str, dict]:
    """Scan all runs and pick the best per modality (highest PSNR)."""
    best: dict[str, dict] = {}

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        result = _parse_run_dir(run_dir)
        if result is None:
            continue
        key, info = result
        if key not in best or info["psnr"] > best[key]["psnr"]:
            best[key] = info

    return best


def copy_results(best: dict[str, dict]) -> dict:
    """Copy best-run images to static/ and build summary dict."""
    summary = {}

    for key, info in sorted(best.items()):
        dest_dir = OUT_DIR / key
        dest_dir.mkdir(parents=True, exist_ok=True)

        src_dir = Path(info["images_dir"])
        for img_name in ["x_true.png", "y.png", "x_hat.png", "comparison.png"]:
            shutil.copy2(src_dir / img_name, dest_dir / img_name)

        summary[key] = {
            "psnr": info["psnr"],
            "ssim": info["ssim"],
            "nrmse": info["nrmse"],
            "solver": info["solver"],
            "images": {
                "x_true": f"/static/img/results/{key}/x_true.png",
                "y": f"/static/img/results/{key}/y.png",
                "x_hat": f"/static/img/results/{key}/x_hat.png",
                "comparison": f"/static/img/results/{key}/comparison.png",
            },
        }

    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning runs...")
    best = collect_best_runs()
    print(f"Found best runs for {len(best)}/{len(VALID_KEYS)} modalities")

    missing = VALID_KEYS - set(best.keys())
    if missing:
        print(f"  Missing modalities: {sorted(missing)}")

    print("Copying images...")
    summary = copy_results(best)

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Done — {len(summary)} modalities with results in {OUT_DIR}")


if __name__ == "__main__":
    main()
