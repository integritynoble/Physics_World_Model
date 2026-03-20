#!/usr/bin/env python3
"""Upload benchmark dataset artifacts to GCS.

Uploads per-modality artifacts (spec.json, true_spec.json, images/) to:
  gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/{tier}/

Also uploads state.md to:
  gs://pwm-benchmark-datasets/datasets/Benchmark/state.md

Usage:
    python3 upload_datasets_to_gcs.py                    # Upload all modalities
    python3 upload_datasets_to_gcs.py --modality ct,mri  # Upload specific modalities
    python3 upload_datasets_to_gcs.py --dry-run           # Preview without uploading
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_BUCKET = "pwm-benchmark-datasets"
GCS_PREFIX = "datasets/Benchmark"

SKIP_DIRS = {"__pycache__", "challenge-data", "gallery", ".ipynb_checkpoints"}
TIERS = ["public", "dev", "hidden"]


def collect_files(modalities: list[str] | None = None) -> list[tuple[Path, str]]:
    """Collect all files to upload."""
    files = []

    # Discover modalities
    if modalities:
        mod_dirs = [BENCHMARK_DIR / m for m in modalities if (BENCHMARK_DIR / m).is_dir()]
    else:
        mod_dirs = [d for d in sorted(BENCHMARK_DIR.iterdir())
                    if d.is_dir() and d.name not in SKIP_DIRS]

    for mod_dir in mod_dirs:
        mod_name = mod_dir.name
        for tier in TIERS:
            tier_dir = mod_dir / tier
            if not tier_dir.is_dir():
                continue

            # spec.json
            spec_file = tier_dir / "spec.json"
            if spec_file.exists():
                files.append((spec_file, f"{GCS_PREFIX}/{mod_name}/{tier}/spec.json"))

            # true_spec.json
            true_spec_file = tier_dir / "true_spec.json"
            if true_spec_file.exists():
                files.append((true_spec_file, f"{GCS_PREFIX}/{mod_name}/{tier}/true_spec.json"))

            # images directory
            images_dir = tier_dir / "images"
            if images_dir.is_dir():
                for img_file in sorted(images_dir.rglob("*")):
                    if img_file.is_file():
                        rel = img_file.relative_to(tier_dir)
                        files.append((img_file, f"{GCS_PREFIX}/{mod_name}/{tier}/{rel}"))

    # state.md
    state_file = BENCHMARK_DIR / "state.md"
    if state_file.exists():
        files.append((state_file, f"{GCS_PREFIX}/state.md"))

    return files


def main():
    parser = argparse.ArgumentParser(description="Upload benchmark datasets to GCS")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET)
    parser.add_argument("--modality", type=str, default="", help="Comma-separated modalities")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    modalities = args.modality.split(",") if args.modality else None
    files_to_upload = collect_files(modalities)

    if not files_to_upload:
        print("No files found to upload.")
        return 1

    total_size = sum(f.stat().st_size for f, _ in files_to_upload)
    print(f"Found {len(files_to_upload)} files ({total_size / 1024 / 1024:.1f} MB) "
          f"to upload to gs://{args.bucket}/{GCS_PREFIX}/")

    if args.dry_run:
        for local_path, gcs_key in files_to_upload[:30]:
            size_kb = local_path.stat().st_size / 1024
            print(f"  {gcs_key} ({size_kb:.1f} KB)")
        if len(files_to_upload) > 30:
            print(f"  ... and {len(files_to_upload) - 30} more")
        return 0

    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        print("ERROR: google-cloud-storage not installed. Run: pip install google-cloud-storage")
        return 1

    client = gcs_storage.Client()
    bucket = client.bucket(args.bucket)

    uploaded = 0
    failed = 0
    for local_path, gcs_key in files_to_upload:
        try:
            blob = bucket.blob(gcs_key)
            blob.upload_from_filename(str(local_path))
            uploaded += 1
            if uploaded % 100 == 0:
                print(f"  Progress: {uploaded}/{len(files_to_upload)}")
        except Exception as e:
            print(f"  FAILED: {gcs_key} — {e}")
            failed += 1

    print(f"\nDone. Uploaded: {uploaded}, Failed: {failed}")
    print(f"Files at: gs://{args.bucket}/{GCS_PREFIX}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
