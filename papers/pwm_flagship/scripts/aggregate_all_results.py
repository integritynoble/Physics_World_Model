#!/usr/bin/env python3
"""Aggregate results from all modality multi-phantom experiments.

Produces a combined JSON summary with per-modality statistics and
confidence intervals for the Nature paper Extended Data tables.

Usage:
    python aggregate_all_results.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_BASE = PROJECT_ROOT / "papers" / "pwm_flagship" / "results"
OUT_DIR = RESULTS_BASE / "combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        logger.warning(f"  Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def extract_modality_summary(data: dict, modality: str) -> dict | None:
    """Extract key metrics from a multiphantom results file."""
    if data is None:
        return None

    # Find the aggregate section (different keys per modality)
    agg = data.get("aggregate", {})
    per_key = None
    for k in agg:
        if isinstance(agg[k], list):
            per_key = k
            break
    if per_key is None:
        return None

    entries = agg[per_key]
    summary = {
        "modality": modality,
        "carrier": data.get("key_findings", {}).get("carrier", "unknown"),
        "gate3_parameter": data.get("key_findings", {}).get("gate3_parameter", "unknown"),
        "n_phantoms": data.get("n_phantoms", 1),
        "mismatch_levels": [],
    }

    for entry in entries:
        # Find the mismatch level key
        level_key = None
        for k in entry:
            if "error" in k.lower() or "offset" in k.lower():
                level_key = k
                break
        if level_key is None:
            continue

        level = {
            "mismatch_value": entry[level_key],
            "mean_delta_psnr_db": entry.get("mean_delta_psnr"),
            "std_delta_psnr_db": entry.get("std_delta_psnr"),
            "ci95_delta_psnr": entry.get("ci95_delta_psnr"),
            "mean_recovery": entry.get("mean_recovery") or entry.get("mean_recovery_ratio"),
            "ci95_recovery": entry.get("ci95_recovery") or entry.get("ci95_recovery_ratio"),
        }
        summary["mismatch_levels"].append(level)

    return summary


def main():
    logger.info("=" * 70)
    logger.info("AGGREGATING ALL MODALITY RESULTS")
    logger.info("=" * 70)

    # Define result file paths
    result_files = {
        "cryo_em": RESULTS_BASE / "real_data_4scenario" / "cryoem_multiphantom_results.json",
        "cbct": RESULTS_BASE / "real_data_4scenario" / "cbct_multiphantom_results.json",
        "ultrasound": RESULTS_BASE / "real_data_4scenario" / "ultrasound_4scenario_results.json",
        "compressive_holography": RESULTS_BASE / "real_data_4scenario" / "compholo_multiphantom_results.json",
        "fluorescence": RESULTS_BASE / "fluorescence_4scenario" / "fluorescence_multiphantom_results.json",
    }

    combined = {
        "description": "Combined multi-phantom 4-scenario results for PWM Nature paper",
        "modalities": {},
    }

    for modality, path in result_files.items():
        logger.info(f"\n  Loading {modality}...")
        data = load_json(path)
        summary = extract_modality_summary(data, modality)
        if summary is not None:
            combined["modalities"][modality] = summary
            n_levels = len(summary["mismatch_levels"])
            if n_levels > 0:
                max_delta = max(l["mean_delta_psnr_db"] or 0
                                for l in summary["mismatch_levels"])
                logger.info(f"    Carrier: {summary['carrier']}, "
                            f"Gate3: {summary['gate3_parameter']}, "
                            f"Phantoms: {summary['n_phantoms']}, "
                            f"Levels: {n_levels}, "
                            f"Max delta: {max_delta:.3f} dB")
            else:
                logger.info(f"    No mismatch levels found")
        else:
            logger.warning(f"    Could not extract summary for {modality}")

    # Generate paper-ready table
    logger.info("\n" + "=" * 90)
    logger.info("PAPER TABLE: Gate 3 Validation Across 5 Carriers (Multi-Phantom)")
    logger.info("=" * 90)
    logger.info(f"{'Modality':<25s} {'Carrier':<12s} {'Gate 3':<18s} "
                f"{'N':<4s} {'Max Δ (dB)':<12s} {'Max Recovery':<12s}")
    logger.info("-" * 90)

    for mod_name, mod_data in combined["modalities"].items():
        levels = mod_data["mismatch_levels"]
        if not levels:
            continue
        max_delta = max(l["mean_delta_psnr_db"] or 0 for l in levels)
        recoveries = [l["mean_recovery"] for l in levels
                      if l["mean_recovery"] is not None]
        # Use the recovery at the largest mismatch level
        max_recovery = levels[-1].get("mean_recovery")
        if max_recovery is None:
            max_recovery_str = "N/A"
        elif isinstance(max_recovery, (int, float)):
            max_recovery_str = f"{max_recovery:.3f}"
        else:
            max_recovery_str = str(max_recovery)

        logger.info(f"{mod_name:<25s} {mod_data['carrier']:<12s} "
                    f"{mod_data['gate3_parameter']:<18s} "
                    f"{mod_data['n_phantoms']:<4d} "
                    f"{max_delta:<12.3f} {max_recovery_str:<12s}")

    logger.info("=" * 90)

    # Save combined
    out_path = OUT_DIR / "combined_multiphantom_summary.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
