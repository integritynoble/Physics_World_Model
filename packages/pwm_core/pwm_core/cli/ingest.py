"""pwm_core.cli.ingest

``pwm ingest`` — ingest a real dataset and register it in the dataset catalog.

Usage
-----
    pwm ingest --modality ct --source /path/to/raw/ --format dicom
    pwm ingest --modality mri --source gs://mybucket/mri_raw/ --format nifti

This command reads raw imaging data, converts it to the PWM canonical format
(numpy arrays with metadata), and registers the dataset in ``dataset_registry.yaml``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ["dicom", "nifti", "npy", "npz", "png", "tiff", "hdf5"]


def add_ingest_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ingest",
        help="Ingest a real dataset into the PWM canonical format",
    )
    p.add_argument("--modality", required=True, help="Physics modality")
    p.add_argument("--source", required=True, help="Path or GCS URI to raw data")
    p.add_argument(
        "--format",
        choices=SUPPORTED_FORMATS,
        default="npy",
        help="Source data format",
    )
    p.add_argument("--out", type=str, default=None, help="Output directory")
    p.add_argument("--tier", choices=["public", "dev", "hidden"], default="public")
    p.add_argument("--upload-gcs", action="store_true", help="Upload canonical data to GCS")
    p.set_defaults(func=cmd_ingest)


def cmd_ingest(args: argparse.Namespace) -> int:
    modality = args.modality
    source = args.source
    fmt = args.format
    tier = args.tier
    out_dir = Path(args.out) if args.out else Path("datasets") / modality / tier / "raw_ingested"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pwm ingest] modality={modality} source={source} format={fmt} out={out_dir}")

    try:
        from pwm_core.data.ingest import ingest_dataset
        n = ingest_dataset(
            modality=modality,
            source=source,
            fmt=fmt,
            out_dir=out_dir,
        )
        print(f"[pwm ingest] Ingested {n} samples to {out_dir}")
    except ImportError:
        logger.warning("pwm_core.data.ingest not available.")
        print("[pwm ingest] Data ingest module not available. Install pwm-core with data extras.")
        return 1

    if args.upload_gcs:
        try:
            from platform.scripts.upload_gallery_to_gcs import upload_directory
            upload_directory(out_dir, f"datasets/Benchmark/{modality}/{tier}/")
            print("[pwm ingest] Uploaded to GCS.")
        except Exception as exc:
            logger.warning("GCS upload failed: %s", exc)
            print(f"[pwm ingest] GCS upload failed: {exc}")

    return 0
