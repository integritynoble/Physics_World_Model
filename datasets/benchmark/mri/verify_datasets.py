#!/usr/bin/env python3
"""Verify 3-tier MRI benchmark HDF5 files (public / dev / hidden)."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

TIERS = {
    "public": {"path": SCRIPT_DIR / "public" / "mri_challenge_public.h5", "n_samples": 11},
    "dev":    {"path": SCRIPT_DIR / "dev" / "mri_challenge_dev.h5",       "n_samples": 20},
    "hidden": {"path": SCRIPT_DIR / "hidden" / "mri_challenge_hidden.h5", "n_samples": 20},
}

EXPECTED_KEYS = ["x_true", "y_kspace", "mask", "coil_maps", "B0_map", "warp_field"]

EXPECTED_SHAPES = {
    "x_true":     (320, 320),
    "y_kspace":   (15, 320, 320),
    "mask":       (320,),
    "coil_maps":  (15, 320, 320),
    "B0_map":     (320, 320),
    "warp_field": (2, 320, 320),
}

EXPECTED_DTYPES = {
    "x_true":     np.float32,
    "y_kspace":   np.complex64,
    "mask":       np.uint8,
    "coil_maps":  np.complex64,
    "B0_map":     np.float32,
    "warp_field": np.float32,
}

METADATA_ATTRS = ["metadata", "spec_ranges", "true_spec"]


def _check(label: str, condition: bool, errors: list[str]) -> bool:
    if condition:
        print(f"    [PASS] {label}")
    else:
        print(f"    [FAIL] {label}")
        errors.append(label)
    return condition


def verify_tier(name: str, h5_path: Path, expected_n: int) -> tuple[bool, dict]:
    """Return (all_passed, summary_dict) for one tier."""
    print(f"\n{'=' * 60}")
    print(f"Tier: {name}  |  {h5_path}")
    print(f"{'=' * 60}")

    errors: list[str] = []
    stats: dict[str, list] = {"x_true_range": [], "B0_range": []}

    if not h5_path.exists():
        print(f"  [FAIL] File not found: {h5_path}")
        return False, {"n_samples": 0, "passed": False}

    with h5py.File(h5_path, "r") as f:
        sample_keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        _check(f"sample count == {expected_n}", len(sample_keys) == expected_n, errors)

        for skey in sample_keys:
            print(f"\n  --- {skey} ---")
            grp = f[skey]

            # 1. Dataset keys
            for dkey in EXPECTED_KEYS:
                _check(f"{dkey} exists", dkey in grp, errors)

            present_keys = [k for k in EXPECTED_KEYS if k in grp]

            # 2. Shapes
            for dkey in present_keys:
                actual = grp[dkey].shape
                expected = EXPECTED_SHAPES[dkey]
                _check(f"{dkey} shape {actual} == {expected}", actual == expected, errors)

            # 3. Dtypes
            for dkey in present_keys:
                actual = grp[dkey].dtype
                expected = EXPECTED_DTYPES[dkey]
                _check(f"{dkey} dtype {actual} == {expected}", actual == expected, errors)

            # 4. Value ranges
            if "x_true" in grp:
                arr = grp["x_true"][:]
                lo, hi = float(arr.min()), float(arr.max())
                _check(f"x_true range [{lo:.4f}, {hi:.4f}] in [0, 1]", lo >= 0.0 and hi <= 1.0, errors)
                stats["x_true_range"].append((lo, hi))

            if "mask" in grp:
                uniq = set(np.unique(grp["mask"][:]).tolist())
                _check(f"mask values {uniq} subset of {{0, 1}}", uniq <= {0, 1}, errors)

            if "B0_map" in grp:
                arr = grp["B0_map"][:]
                lo, hi = float(arr.min()), float(arr.max())
                _check(f"B0_map range [{lo:.4f}, {hi:.4f}] in [-1, 1]", lo >= -1.0 and hi <= 1.0, errors)
                stats["B0_range"].append((lo, hi))

            # 5. Metadata attributes
            for attr_name in METADATA_ATTRS:
                has_attr = attr_name in grp.attrs
                _check(f"attr '{attr_name}' exists", has_attr, errors)
                if has_attr:
                    try:
                        json.loads(grp.attrs[attr_name])
                        _check(f"attr '{attr_name}' valid JSON", True, errors)
                    except (json.JSONDecodeError, TypeError):
                        _check(f"attr '{attr_name}' valid JSON", False, errors)

    passed = len(errors) == 0
    summary = {
        "n_samples": len(sample_keys) if h5_path.exists() else 0,
        "passed": passed,
        "n_errors": len(errors),
        "x_true_range": (
            f"[{min(lo for lo, _ in stats['x_true_range']):.4f}, "
            f"{max(hi for _, hi in stats['x_true_range']):.4f}]"
        ) if stats["x_true_range"] else "N/A",
        "B0_range": (
            f"[{min(lo for lo, _ in stats['B0_range']):.4f}, "
            f"{max(hi for _, hi in stats['B0_range']):.4f}]"
        ) if stats["B0_range"] else "N/A",
    }
    return passed, summary


def print_summary(results: dict[str, dict]) -> None:
    print(f"\n{'=' * 60}")
    print("AGGREGATE SUMMARY")
    print(f"{'=' * 60}")
    header = f"{'Tier':<10} {'Samples':>8} {'Status':>10} {'Errors':>8} {'x_true range':>20} {'B0 range':>20}"
    print(header)
    print("-" * len(header))
    for tier, info in results.items():
        status = "PASS" if info["passed"] else "FAIL"
        print(
            f"{tier:<10} {info['n_samples']:>8} {status:>10} "
            f"{info.get('n_errors', '?'):>8} {info.get('x_true_range', 'N/A'):>20} "
            f"{info.get('B0_range', 'N/A'):>20}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MRI benchmark HDF5 datasets.")
    parser.add_argument(
        "--tier",
        choices=["public", "dev", "hidden", "all"],
        default="all",
        help="Which tier to verify (default: all)",
    )
    args = parser.parse_args()

    tiers_to_check = list(TIERS.keys()) if args.tier == "all" else [args.tier]

    all_passed = True
    results: dict[str, dict] = {}
    for tier_name in tiers_to_check:
        cfg = TIERS[tier_name]
        passed, summary = verify_tier(tier_name, cfg["path"], cfg["n_samples"])
        results[tier_name] = summary
        if not passed:
            all_passed = False

    print_summary(results)

    if all_passed:
        print("\nAll checks PASSED.")
        sys.exit(0)
    else:
        print("\nSome checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
