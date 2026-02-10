"""pwm_core.io.retrieval

Dataset retrieval with SHA256 integrity verification.

Provides utilities for verifying file integrity and copying dataset
files to a local cache directory with checksum verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Dict, List, Optional


def compute_sha256(path: str) -> str:
    """Compute SHA256 hex digest of a file.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    str
        Hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: str, expected_sha256: str) -> bool:
    """Verify file integrity against expected SHA256 hash.

    Parameters
    ----------
    path : str
        Path to the file to verify.
    expected_sha256 : str
        Expected SHA256 hex digest.

    Returns
    -------
    bool
        True if hash matches, False otherwise.
    """
    if not os.path.exists(path):
        return False
    actual = compute_sha256(path)
    return actual.lower() == expected_sha256.lower()


def retrieve_dataset(
    manifest_path: str,
    cache_dir: str,
    verify: bool = True,
) -> str:
    """Load manifest, copy/verify all sample files to cache_dir.

    Parameters
    ----------
    manifest_path : str
        Path to the dataset manifest JSON file.
    cache_dir : str
        Directory to copy dataset files into.
    verify : bool
        If True, verify SHA256 checksums when available.

    Returns
    -------
    str
        Path to the cache directory containing retrieved files.

    Raises
    ------
    FileNotFoundError
        If manifest or source files are not found.
    ValueError
        If SHA256 verification fails.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    os.makedirs(cache_dir, exist_ok=True)

    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    samples = manifest.get("samples", [])

    for sample in samples:
        src_path = sample.get("path", "")
        if not os.path.isabs(src_path):
            src_path = os.path.join(manifest_dir, src_path)

        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Sample file not found: {src_path}")

        dst_path = os.path.join(cache_dir, os.path.basename(src_path))

        # Copy if not already cached
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

        # Verify if checksum provided
        if verify and "sha256" in sample:
            if not verify_file(dst_path, sample["sha256"]):
                raise ValueError(
                    f"SHA256 mismatch for {dst_path}: "
                    f"expected {sample['sha256']}, "
                    f"got {compute_sha256(dst_path)}"
                )

    return cache_dir
