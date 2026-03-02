#!/usr/bin/env python3
"""Upload 3-tier MRI benchmark HDF5 files (and metadata) to Google Cloud Storage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Import GCSDatasetStore from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.datasets.gcs_store import GCSDatasetStore  # noqa: E402

TIERS = ("public", "dev", "hidden")
GCS_PREFIX = "challenge-data/mri"
HERE = Path(__file__).resolve().parent


def _collect_files(tier: str) -> List[Tuple[Path, str]]:
    """Return a list of (local_path, gcs_key) pairs for a single tier."""
    tier_dir = HERE / tier
    pairs: List[Tuple[Path, str]] = []

    # 1. Main HDF5 file
    h5 = tier_dir / f"mri_challenge_{tier}.h5"
    if h5.exists():
        pairs.append((h5, f"{GCS_PREFIX}/mri_challenge_{tier}.h5"))
    else:
        print(f"  WARNING: {h5} not found -- skipping H5 for tier '{tier}'")

    # 2. README.md (optional)
    readme = tier_dir / "README.md"
    if readme.exists():
        pairs.append((readme, f"{GCS_PREFIX}/{tier}_README.md"))

    # 3. spec.json files inside images/ subdirectories
    images_dir = tier_dir / "images"
    if images_dir.is_dir():
        for spec in sorted(images_dir.rglob("spec.json")):
            rel = spec.relative_to(tier_dir)  # e.g. images/sample_00_.../spec.json
            pairs.append((spec, f"{GCS_PREFIX}/{tier}/{rel}"))

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload MRI benchmark datasets to Google Cloud Storage.",
    )
    parser.add_argument(
        "--tier",
        choices=[*TIERS, "all"],
        default="all",
        help="Which tier(s) to upload (default: all).",
    )
    parser.add_argument(
        "--bucket",
        default="pwm-benchmark-datasets",
        help="GCS bucket name (default: pwm-benchmark-datasets).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading.",
    )
    args = parser.parse_args()

    tiers = list(TIERS) if args.tier == "all" else [args.tier]
    store = GCSDatasetStore(bucket_name=args.bucket)

    # ---- Credential check (skip for dry-run) -----------------------------
    if not args.dry_run and not store.available:
        print(
            "ERROR: GCS credentials are not configured.\n"
            "Set GOOGLE_APPLICATION_CREDENTIALS or run:\n"
            "  gcloud auth application-default login",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Collect all files ------------------------------------------------
    all_files: List[Tuple[str, Path, str]] = []  # (tier, local, key)
    for tier in tiers:
        for local, key in _collect_files(tier):
            all_files.append((tier, local, key))

    if not all_files:
        print("Nothing to upload.")
        return

    # ---- Dry-run report ---------------------------------------------------
    if args.dry_run:
        print(f"[DRY RUN] Would upload {len(all_files)} file(s) to gs://{args.bucket}/\n")
        for tier, local, key in all_files:
            size_kb = local.stat().st_size / 1024
            print(f"  [{tier:>6}] {local.name:40s} -> {key}  ({size_kb:,.1f} KB)")
        return

    # ---- Upload -----------------------------------------------------------
    uploaded: List[str] = []
    failed: List[str] = []

    for tier, local, key in all_files:
        print(f"  [{tier:>6}] Uploading {local.name} ...")
        uri = store.upload(local, key)
        if uri is None:
            print(f"         FAILED: {key}")
            failed.append(key)
            continue

        # Verify the object landed
        if store.exists(key):
            print(f"         OK: {uri}")
            uploaded.append(uri)
        else:
            print(f"         WARN: upload returned URI but exists() is False: {key}")
            failed.append(key)

    # ---- Summary ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Upload complete: {len(uploaded)} succeeded, {len(failed)} failed")
    if uploaded:
        print("\nUploaded:")
        for uri in uploaded:
            print(f"  {uri}")
    if failed:
        print("\nFailed:")
        for key in failed:
            print(f"  {key}")
        sys.exit(1)


if __name__ == "__main__":
    main()
