"""Upload all standard datasets to Google Cloud Storage.

Usage:
    1. First authenticate: gcloud auth login
    2. Then run: python scripts/upload_standard_to_gcs.py

Uploads to: gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/
Total size: ~1 GB across 170 modalities.
"""
import subprocess
import sys
from pathlib import Path

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
GCS_BUCKET = "gs://pwm-benchmark-datasets/datasets/Benchmark"

def upload_modality(mod_name):
    """Upload a single modality's standard dataset to GCS."""
    local_dir = BASE / mod_name / "standard"
    if not local_dir.exists():
        return False

    gcs_path = f"{GCS_BUCKET}/{mod_name}/standard/"

    # Upload HDF5 files
    h5_files = sorted(local_dir.glob("*.h5"))
    json_files = sorted(local_dir.glob("*.json"))
    img_dir = local_dir / "images"

    all_files = list(h5_files) + list(json_files)
    if img_dir.exists():
        all_files += sorted(img_dir.glob("*.png"))

    if not all_files:
        return False

    # Use gcloud storage cp for each file
    for f in all_files:
        rel = f.relative_to(local_dir)
        dst = f"{gcs_path}{str(rel).replace(chr(92), '/')}"
        cmd = ["gcloud", "storage", "cp", str(f), dst]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  ERROR uploading {rel}: {result.stderr[:100]}")
            return False

    return True


def main():
    # Check gcloud auth
    result = subprocess.run(["gcloud", "auth", "print-access-token"],
                          capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        print("ERROR: Not authenticated with GCS.")
        print("Please run: gcloud auth login")
        sys.exit(1)

    print(f"Uploading standard datasets to {GCS_BUCKET}/")
    print("=" * 60)

    modalities = sorted([
        d.name for d in BASE.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (d / "standard").exists()
    ])

    success = 0
    failed = 0
    for i, mod in enumerate(modalities):
        n_files = len(list((BASE / mod / "standard").glob("*.h5")))
        print(f"[{i+1}/{len(modalities)}] {mod} ({n_files} h5 files)...", end=" ", flush=True)
        try:
            if upload_modality(mod):
                print("OK")
                success += 1
            else:
                print("SKIP (no files)")
        except Exception as e:
            print(f"FAILED: {str(e)[:60]}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Upload complete: {success} success, {failed} failed, {len(modalities)} total")
    print(f"GCS path: {GCS_BUCKET}/{{modality}}/standard/")


if __name__ == "__main__":
    main()
