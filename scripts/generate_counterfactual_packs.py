#!/usr/bin/env python3
"""Generate counterfactual packs for LIP-Arena validation.

Usage:
    python scripts/generate_counterfactual_packs.py --modality cassi --out-dir /tmp/cfpacks
    python scripts/generate_counterfactual_packs.py --modality spc --split probe
    python scripts/generate_counterfactual_packs.py --modality all
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure pwm_core is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "pwm_core"))

from pwm_core.counterfactual.cassi_generator import CassiCounterfactualGenerator
from pwm_core.counterfactual.cacti_generator import CactiCounterfactualGenerator
from pwm_core.counterfactual.spc_generator import SpcCounterfactualGenerator
from pwm_core.counterfactual.schema import SplitKind

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

GENERATORS = {
    "cassi": CassiCounterfactualGenerator,
    "spc": SpcCounterfactualGenerator,
    "cacti": CactiCounterfactualGenerator,
}

DEFAULT_OUT_DIR = REPO_ROOT / "contrib" / "counterfactual_packs"


def main():
    parser = argparse.ArgumentParser(
        description="Generate counterfactual packs for LIP-Arena validation."
    )
    parser.add_argument(
        "--modality",
        choices=["cassi", "spc", "cacti", "all"],
        default="all",
        help="Which modality pack to generate (default: all).",
    )
    parser.add_argument(
        "--split",
        choices=["probe", "hidden", "both"],
        default="both",
        help="Which split to generate (default: both).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--seed-public",
        type=int,
        default=2026_02_18,
        help="Seed for probe split (default: 20260218).",
    )
    parser.add_argument(
        "--seed-hidden",
        type=int,
        default=9999_02_18,
        help="Seed for hidden split (default: 99990218).",
    )
    args = parser.parse_args()

    modalities = list(GENERATORS.keys()) if args.modality == "all" else [args.modality]

    for mod in modalities:
        logger.info("=" * 60)
        logger.info("Generating %s counterfactual pack", mod.upper())
        logger.info("=" * 60)

        gen_cls = GENERATORS[mod]
        gen = gen_cls(seed_public=args.seed_public, seed_hidden=args.seed_hidden)
        pack_dir = args.out_dir / f"{mod}_cfpack_v1"

        t0 = time.time()
        try:
            manifest = gen.generate_pack(pack_dir)
            elapsed = time.time() - t0
            logger.info(
                "%s pack complete: %d scenarios in %.1f s -> %s",
                mod.upper(),
                manifest.n_scenarios,
                elapsed,
                pack_dir,
            )
        except FileNotFoundError as e:
            logger.error("Dataset not found for %s: %s", mod, e)
            logger.error("Skipping %s (ensure source datasets are available).", mod)
            continue
        except Exception:
            logger.exception("Failed to generate %s pack", mod)
            raise

    logger.info("Done.")


if __name__ == "__main__":
    main()
