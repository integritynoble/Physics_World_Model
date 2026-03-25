"""pwm_core.cli.synthesize

``pwm synthesize`` — generate synthetic benchmark data for a modality.

Usage
-----
    pwm synthesize --modality ct --n 20 --tier public --out datasets/ct/

This command generates synthetic (modality, measurement, ground-truth) triples
using the universal dataset generator and writes them to the output directory.
The split seed scheme (public=0, dev=+10000, hidden=+20000) is applied
automatically so each tier uses different ground-truth data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SEED_OFFSETS = {"public": 0, "dev": 10000, "hidden": 20000}


def add_synthesize_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "synthesize",
        help="Generate synthetic benchmark data for a modality",
    )
    p.add_argument("--modality", required=True, help="Physics modality (e.g. ct, cassi)")
    p.add_argument("--n", type=int, default=20, help="Number of samples to generate")
    p.add_argument(
        "--tier",
        choices=["public", "dev", "hidden"],
        default="public",
        help="Dataset split tier",
    )
    p.add_argument("--out", type=str, default=None, help="Output directory (default: datasets/{modality}/)")
    p.add_argument("--seed", type=int, default=None, help="Base random seed (default: tier offset)")
    p.add_argument("--upload-gcs", action="store_true", help="Upload to GCS after generation")
    p.set_defaults(func=cmd_synthesize)


def cmd_synthesize(args: argparse.Namespace) -> int:
    modality = args.modality
    n = args.n
    tier = args.tier
    seed = args.seed if args.seed is not None else SEED_OFFSETS[tier]
    out_dir = Path(args.out) if args.out else Path("datasets") / modality / tier
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pwm synthesize] modality={modality} tier={tier} n={n} seed={seed} out={out_dir}")

    try:
        from pwm_core.data.generator import generate_benchmark_split
        generate_benchmark_split(
            modality=modality,
            n_samples=n,
            tier=tier,
            base_seed=seed,
            out_dir=out_dir,
        )
        print(f"[pwm synthesize] Done. {n} samples written to {out_dir}")
    except ImportError:
        logger.warning("pwm_core.data.generator not available; skipping generation.")
        print("[pwm synthesize] Generator not available. Install pwm-core with data extras.")
        return 1

    if args.upload_gcs:
        try:
            from platform.scripts.upload_gallery_to_gcs import upload_directory
            upload_directory(out_dir, f"datasets/Benchmark/{modality}/{tier}/")
            print("[pwm synthesize] Uploaded to GCS.")
        except Exception as exc:
            logger.warning("GCS upload failed: %s", exc)
            print(f"[pwm synthesize] GCS upload failed: {exc}")

    return 0
