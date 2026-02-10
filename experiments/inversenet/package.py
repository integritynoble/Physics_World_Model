"""Package InverseNet dataset for HuggingFace / Zenodo distribution.

Creates a ``tar.gz`` archive with SHA-256 checksums for every file,
plus a top-level ``checksums.sha256`` manifest.

Usage::

    python -m experiments.inversenet.package \\
        --data_dirs datasets/inversenet_spc datasets/inversenet_cacti datasets/inversenet_cassi \\
        --out inversenet_v0.1.0.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tarfile
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1 << 16  # 64 KiB


def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_files(data_dir: str) -> List[str]:
    """Recursively collect all files under data_dir (relative paths)."""
    files = []
    for root, _dirs, fnames in os.walk(data_dir):
        for fn in sorted(fnames):
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, start=os.path.dirname(data_dir))
            files.append((full, rel))
    return files


def package_dataset(
    data_dirs: List[str],
    out_path: str,
    dataset_card_path: Optional[str] = None,
) -> Dict[str, str]:
    """Package dataset directories into a tar.gz with checksums.

    Returns:
        Dict mapping relative path -> sha256 hex digest.
    """
    checksums: Dict[str, str] = {}
    all_files: List[tuple] = []

    for d in data_dirs:
        if not os.path.isdir(d):
            logger.warning(f"Skipping missing directory: {d}")
            continue
        all_files.extend(collect_files(d))

    # Include dataset card if provided
    if dataset_card_path and os.path.exists(dataset_card_path):
        all_files.append((dataset_card_path, "dataset_card.md"))

    logger.info(f"Packaging {len(all_files)} files -> {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Build checksum manifest
    for full_path, rel_path in all_files:
        checksums[rel_path] = sha256_file(full_path)

    # Write checksums file to a temp location
    checksum_tmp = out_path + ".checksums.tmp"
    with open(checksum_tmp, "w") as f:
        for rel, digest in sorted(checksums.items()):
            f.write(f"{digest}  {rel}\n")

    # Create tar.gz
    with tarfile.open(out_path, "w:gz") as tar:
        # Add checksum manifest first
        tar.add(checksum_tmp, arcname="checksums.sha256")

        # Add all data files
        for full_path, rel_path in all_files:
            tar.add(full_path, arcname=rel_path)

    # Clean up
    if os.path.exists(checksum_tmp):
        os.remove(checksum_tmp)

    logger.info(
        f"Archive created: {out_path} "
        f"({os.path.getsize(out_path) / 1e6:.1f} MB, "
        f"{len(checksums)} files)"
    )
    return checksums


def verify_archive(tar_path: str) -> bool:
    """Verify that all files in the archive match their checksums."""
    import io

    with tarfile.open(tar_path, "r:gz") as tar:
        # Read checksum manifest
        checksums_member = tar.getmember("checksums.sha256")
        checksums_data = tar.extractfile(checksums_member).read().decode()
        expected: Dict[str, str] = {}
        for line in checksums_data.strip().split("\n"):
            if not line:
                continue
            digest, rel = line.split("  ", 1)
            expected[rel] = digest

        # Verify each file
        ok = True
        for member in tar.getmembers():
            if member.name == "checksums.sha256" or member.isdir():
                continue
            if member.name not in expected:
                logger.warning(f"Extra file not in checksums: {member.name}")
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            h = hashlib.sha256()
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                h.update(chunk)
            actual = h.hexdigest()
            if actual != expected[member.name]:
                logger.error(
                    f"MISMATCH {member.name}: "
                    f"expected {expected[member.name]}, got {actual}"
                )
                ok = False

    return ok


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Package InverseNet dataset for distribution"
    )
    parser.add_argument(
        "--data_dirs",
        nargs="*",
        default=[
            "datasets/inversenet_spc",
            "datasets/inversenet_cacti",
            "datasets/inversenet_cassi",
        ],
    )
    parser.add_argument("--out", default="inversenet_v0.1.0.tar.gz")
    parser.add_argument(
        "--dataset_card",
        default="experiments/inversenet/dataset_card.md",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    checksums = package_dataset(args.data_dirs, args.out, args.dataset_card)

    if args.verify:
        ok = verify_archive(args.out)
        if ok:
            logger.info("Verification PASSED")
        else:
            logger.error("Verification FAILED")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
