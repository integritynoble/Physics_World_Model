#!/usr/bin/env python3
"""Strip ground truth (x_true, true_spec) from dev-tier HDF5 files on GCS.

Dev-tier files should NOT contain ground truth — they're for blind evaluation.
This script downloads each dev file from GCS, removes x_true datasets and
true_spec attributes, then re-uploads the stripped file.

Usage:
    python3 scripts/strip_dev_ground_truth.py
    python3 scripts/strip_dev_ground_truth.py --dry-run
    python3 scripts/strip_dev_ground_truth.py --variant ct
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import h5py

logger = logging.getLogger(__name__)

GCS_BUCKET = "pwm-benchmark-datasets"
GCS_PREFIX = "challenge-data/v1.0/"


def strip_dev_file(local_path: Path) -> tuple[int, int]:
    """Strip x_true and true_spec from a dev HDF5 file in-place.

    Returns (x_true_removed, true_spec_removed) counts.
    """
    x_true_count = 0
    true_spec_count = 0

    with h5py.File(local_path, "a") as f:
        for sample_key in sorted(f.keys()):
            grp = f[sample_key]

            if "x_true" in grp:
                del grp["x_true"]
                x_true_count += 1

            if "true_spec" in grp.attrs:
                del grp.attrs["true_spec"]
                true_spec_count += 1

    return x_true_count, true_spec_count


def main():
    parser = argparse.ArgumentParser(
        description="Strip ground truth from dev-tier HDF5 files on GCS"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be modified without changing them",
    )
    parser.add_argument(
        "--variant", default=None,
        help="Process only this variant (e.g., 'ct'). Default: all dev files.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        from google.cloud import storage
    except ImportError:
        logger.error("google-cloud-storage not installed")
        sys.exit(1)

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    # List all dev-tier files
    blobs = list(bucket.list_blobs(prefix=GCS_PREFIX))
    dev_blobs = [b for b in blobs if "_challenge_dev.h5" in b.name]

    if args.variant:
        dev_blobs = [
            b for b in dev_blobs
            if b.name.endswith(f"{args.variant}_challenge_dev.h5")
        ]

    if not dev_blobs:
        logger.info("No dev-tier files found.")
        return

    logger.info("Found %d dev-tier files to process.", len(dev_blobs))

    if args.dry_run:
        for blob in dev_blobs:
            logger.info("  Would strip: %s", blob.name)
        return

    tmpdir = Path(tempfile.mkdtemp(prefix="pwm_strip_"))
    stripped = 0
    skipped = 0
    errors = 0

    try:
        for i, blob in enumerate(dev_blobs, 1):
            fname = blob.name.split("/")[-1]
            local = tmpdir / fname

            try:
                # Download
                blob.download_to_filename(str(local))
                orig_size = local.stat().st_size

                # Strip
                x_count, spec_count = strip_dev_file(local)

                if x_count == 0 and spec_count == 0:
                    skipped += 1
                    local.unlink()
                    continue

                new_size = local.stat().st_size
                saved_kb = (orig_size - new_size) / 1024

                # Re-upload
                blob.upload_from_filename(str(local))
                stripped += 1
                logger.info(
                    "  [%d/%d] %s: removed %d x_true, %d true_spec (saved %.0f KB)",
                    i, len(dev_blobs), fname, x_count, spec_count, saved_kb,
                )

            except Exception as e:
                logger.error("  [%d/%d] %s: ERROR — %s", i, len(dev_blobs), fname, e)
                errors += 1

            finally:
                if local.exists():
                    local.unlink()

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    logger.info(
        "Done. Stripped: %d, Already clean: %d, Errors: %d",
        stripped, skipped, errors,
    )


if __name__ == "__main__":
    main()
