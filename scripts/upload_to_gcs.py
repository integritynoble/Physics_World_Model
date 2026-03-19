#!/usr/bin/env python3
"""Upload heavy datasets and model weights to GCS.

Run this script to upload local datasets to gs://pwm-benchmark-datasets/.
Requires authenticated Google Cloud credentials:
    gcloud auth application-default login

Usage:
    python scripts/upload_to_gcs.py [--what all|benchmark|weights|results|inversenet]
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUCKET_NAME = "pwm-benchmark-datasets"
PROJECT = "subtle-app-431618-i1"


def get_client():
    try:
        from google.cloud import storage
        import google.auth
        creds, project = google.auth.default()
        return storage.Client(credentials=creds, project=project or PROJECT)
    except Exception as e:
        print(f"ERROR: Could not authenticate with GCS: {e}")
        print("Run: gcloud auth application-default login")
        sys.exit(1)


def upload_dir(client, local_dir: Path, gcs_prefix: str, dry_run: bool = False):
    """Upload all files from local_dir to GCS under gcs_prefix."""
    bucket = client.bucket(BUCKET_NAME)
    local_dir = Path(local_dir)
    if not local_dir.exists():
        print(f"  SKIP: {local_dir} does not exist")
        return 0

    count = 0
    total_bytes = 0
    for fpath in sorted(local_dir.rglob("*")):
        if not fpath.is_file():
            continue
        rel = fpath.relative_to(local_dir)
        blob_name = f"{gcs_prefix}/{rel}".replace("\\", "/")
        size_mb = fpath.stat().st_size / 1024 / 1024
        if dry_run:
            print(f"  [DRY RUN] {fpath} -> gs://{BUCKET_NAME}/{blob_name} ({size_mb:.1f} MB)")
        else:
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(fpath))
            print(f"  Uploaded: {blob_name} ({size_mb:.1f} MB)")
        count += 1
        total_bytes += fpath.stat().st_size

    return count


def upload_benchmark(client, dry_run=False):
    """Upload 168-modality benchmark datasets."""
    print("\n=== Uploading benchmark datasets ===")
    bench_dir = ROOT / "datasets" / "benchmark"
    if not bench_dir.exists():
        print(f"  No benchmark data at {bench_dir}")
        return
    n = upload_dir(client, bench_dir, "benchmark", dry_run=dry_run)
    print(f"  Total: {n} files")


def upload_weights(client, dry_run=False):
    """Upload pre-trained model weights."""
    print("\n=== Uploading model weights ===")
    weights_dir = (
        ROOT.parent.parent.parent
        / "PWM4/Physics_World_Model-master/packages/pwm_core/pwm_core/weights"
    )
    if not weights_dir.exists():
        print(f"  No weights at {weights_dir}")
        return
    n = upload_dir(client, weights_dir, "weights", dry_run=dry_run)
    print(f"  Total: {n} files")


def upload_results(client, dry_run=False):
    """Upload paper reconstruction results."""
    print("\n=== Uploading paper results ===")
    dirs = [
        (ROOT / "papers" / "inversenet" / "results", "results/inversenet"),
        (ROOT / "papers" / "pwmi_cassi" / "results", "results/pwmi_cassi"),
        (ROOT / "papers" / "inversenet" / "data" / "spc" / "models",
         "results/inversenet_models"),
    ]
    for local_dir, gcs_prefix in dirs:
        n = upload_dir(client, local_dir, gcs_prefix, dry_run=dry_run)
        print(f"  {local_dir.name}: {n} files")


def upload_inversenet(client, dry_run=False):
    """Upload InverseNet and LIP Arena demo samples."""
    print("\n=== Uploading InverseNet / LIP Arena datasets ===")
    dirs = [
        (ROOT / "datasets" / "inversenet_cacti", "inversenet/cacti"),
        (ROOT / "datasets" / "inversenet_cassi", "inversenet/cassi"),
        (ROOT / "datasets" / "inversenet_spc", "inversenet/spc"),
        (ROOT / "datasets" / "lip_arena", "lip_arena"),
    ]
    for local_dir, gcs_prefix in dirs:
        n = upload_dir(client, local_dir, gcs_prefix, dry_run=dry_run)
        print(f"  {local_dir.name}: {n} files")


def main():
    parser = argparse.ArgumentParser(description="Upload datasets to GCS")
    parser.add_argument(
        "--what",
        choices=["all", "benchmark", "weights", "results", "inversenet"],
        default="all",
        help="What to upload (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    print(f"GCS bucket: gs://{BUCKET_NAME}/")
    if args.dry_run:
        print("DRY RUN - no files will be uploaded")
        client = None
    else:
        client = get_client()

    if args.what in ("all", "benchmark"):
        upload_benchmark(client, dry_run=args.dry_run)
    if args.what in ("all", "weights"):
        upload_weights(client, dry_run=args.dry_run)
    if args.what in ("all", "results"):
        upload_results(client, dry_run=args.dry_run)
    if args.what in ("all", "inversenet"):
        upload_inversenet(client, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
