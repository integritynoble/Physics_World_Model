#!/usr/bin/env python3
"""Upload CASSI checkpoints to GCS at gs://pwm-benchmark-datasets/checkpoint/

Run: python scripts/upload_checkpoints_gcs.py

Requires: gcloud auth login (or application-default credentials)
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent

# All checkpoint files to upload
CHECKPOINTS = {
    # HDNet
    "packages/pwm_core/pwm_core/weights/hdnet/hdnet.pth": "checkpoint/cassi/hdnet.pth",
    # MST-L
    "packages/pwm_core/weights/mst/mst_l.pth": "checkpoint/cassi/mst_l.pth",
    # PnP-HSICNN / HSI-SDeCNN
    "packages/pwm_core/pwm_core/weights/hsi_sdecnn/deep_denoiser.pth": "checkpoint/cassi/deep_denoiser.pth",
}

# CASSI checkpoints from Google Drive (cassi_checkpoints/)
cassi_ckpt_dir = ROOT / "packages" / "pwm_core" / "pwm_core" / "weights" / "cassi_checkpoints"
if cassi_ckpt_dir.exists():
    for subdir in sorted(cassi_ckpt_dir.iterdir()):
        if subdir.is_dir():
            for pth_file in subdir.glob("*.pth"):
                rel = f"checkpoint/cassi/{subdir.name}/{pth_file.name}"
                CHECKPOINTS[str(pth_file.relative_to(ROOT))] = rel

# RED-CNN for CT
redcnn = ROOT / "packages" / "pwm_core" / "pwm_core" / "weights" / "redcnn" / "redcnn.pth"
if redcnn.exists():
    CHECKPOINTS[str(redcnn.relative_to(ROOT))] = "checkpoint/ct/redcnn.pth"

GCS_BUCKET = "gs://pwm-benchmark-datasets"


def upload():
    print("=" * 60)
    print("Uploading checkpoints to GCS")
    print(f"Bucket: {GCS_BUCKET}")
    print("=" * 60)

    success = 0
    failed = 0

    for local_rel, gcs_rel in sorted(CHECKPOINTS.items()):
        local_path = ROOT / local_rel
        if not local_path.exists():
            print(f"  SKIP {local_rel}: not found")
            continue

        gcs_path = f"{GCS_BUCKET}/{gcs_rel}"
        size_mb = local_path.stat().st_size / 1e6
        print(f"  {local_rel} ({size_mb:.1f} MB) -> {gcs_path}")

        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket("pwm-benchmark-datasets")
            blob = bucket.blob(gcs_rel)
            blob.upload_from_filename(str(local_path))
            success += 1
            print(f"    OK")
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

    print(f"\nDone: {success} uploaded, {failed} failed")


if __name__ == "__main__":
    upload()
