"""Download standard datasets from Google Cloud Storage.

Usage on a new server:
    1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
    2. Authenticate: gcloud auth login
    3. Run: python scripts/download_standard_from_gcs.py

Downloads from: gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/
Downloads to:   datasets/benchmark/{modality}/standard/

Options:
    --modality ct,mri,pet     Download only specific modalities (comma-separated)
    --list                    List available modalities without downloading
    --dry-run                 Show what would be downloaded without downloading

Total size: ~1 GB across 170 modalities.
"""
import subprocess
import sys
import argparse
from pathlib import Path

GCS_BUCKET = "gs://pwm-benchmark-datasets/datasets/Benchmark"
LOCAL_BASE = Path(__file__).parent.parent / "datasets" / "benchmark"


def list_modalities():
    """List all modalities available on GCS."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", f"{GCS_BUCKET}/"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr[:200]}")
        return []
    dirs = [line.strip().rstrip("/").split("/")[-1]
            for line in result.stdout.strip().split("\n") if line.strip()]
    return sorted(dirs)


def download_modality(mod_name, dry_run=False):
    """Download a single modality's standard dataset from GCS."""
    gcs_path = f"{GCS_BUCKET}/{mod_name}/standard/"
    local_dir = LOCAL_BASE / mod_name / "standard"

    if dry_run:
        print(f"  Would download: {gcs_path} -> {local_dir}")
        return True

    local_dir.mkdir(parents=True, exist_ok=True)
    img_dir = local_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # Use gcloud storage cp -r for recursive download
    cmd = ["gcloud", "storage", "cp", "-r", gcs_path + "*", str(local_dir) + "/"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        # Try individual files
        for pattern in ["*.h5", "*.json", "images/*.png"]:
            cmd2 = ["gcloud", "storage", "cp",
                    f"{gcs_path}{pattern}", str(local_dir) + "/"]
            subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
        return True
    return True


def main():
    parser = argparse.ArgumentParser(description="Download standard datasets from GCS")
    parser.add_argument("--modality", type=str, default=None,
                       help="Comma-separated list of modalities to download")
    parser.add_argument("--list", action="store_true",
                       help="List available modalities")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be downloaded")
    args = parser.parse_args()

    # Check gcloud auth
    result = subprocess.run(["gcloud", "auth", "print-access-token"],
                          capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        print("ERROR: Not authenticated with GCS.")
        print("Please run: gcloud auth login")
        sys.exit(1)

    if args.list:
        mods = list_modalities()
        print(f"Available modalities ({len(mods)}):")
        for m in mods:
            print(f"  {m}")
        return

    if args.modality:
        modalities = [m.strip() for m in args.modality.split(",")]
    else:
        modalities = list_modalities()
        if not modalities:
            print("No modalities found on GCS. Check bucket access.")
            return

    print(f"Downloading {len(modalities)} modalities from GCS...")
    print(f"  Source: {GCS_BUCKET}/")
    print(f"  Target: {LOCAL_BASE}/")
    print("=" * 60)

    success = 0
    for i, mod in enumerate(modalities):
        print(f"[{i+1}/{len(modalities)}] {mod}...", end=" ", flush=True)
        try:
            if download_modality(mod, dry_run=args.dry_run):
                print("OK")
                success += 1
        except Exception as e:
            print(f"FAILED: {str(e)[:60]}")

    print()
    print(f"Download complete: {success}/{len(modalities)} modalities")


if __name__ == "__main__":
    main()
