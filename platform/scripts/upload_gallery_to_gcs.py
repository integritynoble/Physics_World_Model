#!/usr/bin/env python3
"""Upload pre-computed benchmark gallery to Google Cloud Storage.

Usage:
    python3 scripts/upload_gallery_to_gcs.py
    python3 scripts/upload_gallery_to_gcs.py --bucket my-bucket-name

Uploads:
    - All images from pwm_platform/static/img/benchmark_gallery/
    - benchmark_gallery.json from pwm_platform/static/benchmark-data/

Prerequisites:
    - google-cloud-storage installed
    - GCS credentials configured (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth)
    - Bucket must already exist (or service account must have storage.buckets.create)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_BUCKET = "pwm-benchmark-datasets"
GCS_PREFIX = "benchmark_gallery"


def main():
    parser = argparse.ArgumentParser(description="Upload benchmark gallery to GCS")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET, help="GCS bucket name")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    args = parser.parse_args()

    platform_root = Path(__file__).resolve().parent.parent
    img_root = platform_root / "pwm_platform" / "static" / "img" / "benchmark_gallery"
    json_path = platform_root / "pwm_platform" / "static" / "benchmark-data" / "benchmark_gallery.json"

    # Collect all files to upload
    files_to_upload = []

    # Images
    if img_root.exists():
        for f in sorted(img_root.rglob("*")):
            if f.is_file():
                rel = f.relative_to(platform_root / "pwm_platform" / "static")
                gcs_key = f"{GCS_PREFIX}/{rel}"
                files_to_upload.append((f, gcs_key))

    # JSON
    if json_path.exists():
        files_to_upload.append((json_path, f"{GCS_PREFIX}/benchmark_gallery.json"))

    if not files_to_upload:
        print("No files found to upload.")
        sys.exit(1)

    print(f"Found {len(files_to_upload)} files to upload to gs://{args.bucket}/{GCS_PREFIX}/")

    if args.dry_run:
        for local_path, gcs_key in files_to_upload:
            size_kb = local_path.stat().st_size / 1024
            print(f"  {gcs_key} ({size_kb:.1f} KB)")
        print(f"\nTotal: {sum(f.stat().st_size for f, _ in files_to_upload) / 1024:.1f} KB")
        return

    # Upload
    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        print("ERROR: google-cloud-storage not installed. Run: pip install google-cloud-storage")
        sys.exit(1)

    try:
        client = gcs_storage.Client()
        bucket = client.bucket(args.bucket)

        # Check if bucket exists, create if not
        if not bucket.exists():
            print(f"Bucket {args.bucket} does not exist, creating...")
            bucket = client.create_bucket(args.bucket, location="us-central1")
            print(f"Created bucket {args.bucket}")

    except Exception as e:
        print(f"ERROR: Cannot access GCS bucket '{args.bucket}': {e}")
        print("\nTo fix this, ensure:")
        print("  1. GOOGLE_APPLICATION_CREDENTIALS is set, or gcloud auth is configured")
        print("  2. The service account has Storage Admin or Storage Object Admin role")
        print("  3. The bucket exists or the account can create buckets")
        sys.exit(1)

    uploaded = 0
    failed = 0
    for local_path, gcs_key in files_to_upload:
        try:
            blob = bucket.blob(gcs_key)
            blob.upload_from_filename(str(local_path))
            size_kb = local_path.stat().st_size / 1024
            print(f"  Uploaded: {gcs_key} ({size_kb:.1f} KB)")
            uploaded += 1
        except Exception as e:
            print(f"  FAILED:   {gcs_key} — {e}")
            failed += 1

    print(f"\nDone. Uploaded: {uploaded}, Failed: {failed}")
    if uploaded > 0:
        print(f"Files available at: gs://{args.bucket}/{GCS_PREFIX}/")


if __name__ == "__main__":
    main()
