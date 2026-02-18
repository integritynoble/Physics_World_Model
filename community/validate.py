#!/usr/bin/env python3
"""Validate RunBundle submissions for PWM weekly challenges.

Checks that a submitted zip file conforms to the RunBundle v0.3.0 schema
as specified in docs/contracts/runbundle_schema.md.

Usage:
    python community/validate.py submission.zip
    python community/validate.py submission.zip --verbose
    python community/validate.py /path/to/runbundle_dir  # unzipped directory

Exit codes:
    0 = PASS (all checks pass)
    1 = FAIL (validation errors found)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


RUNBUNDLE_VERSION = "0.3.0"
MANIFEST_FILENAME = "runbundle_manifest.json"

REQUIRED_TOP_LEVEL = {"version", "spec_id", "timestamp", "provenance", "metrics", "artifacts", "hashes"}
REQUIRED_PROVENANCE = {"git_hash", "seeds", "platform", "pwm_version"}
REQUIRED_METRICS = {"psnr_db", "ssim", "runtime_s"}
REQUIRED_ARTIFACTS = {"x_gt", "y", "x_hat"}


class ValidationResult:
    """Collect validation errors and warnings."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = []
        if self.passed:
            lines.append("PASS: RunBundle validation successful.")
        else:
            lines.append(f"FAIL: {len(self.errors)} error(s) found.")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def sha256_of_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_finite_float(value) -> bool:
    """Check that value is a finite float (not NaN, not Inf)."""
    try:
        v = float(value)
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False


def _parse_iso8601(ts: str) -> Optional[datetime]:
    """Attempt to parse an ISO 8601 timestamp."""
    # Try common formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def validate_manifest(manifest: dict, result: ValidationResult) -> None:
    """Validate the manifest dict structure and field values."""
    # Check required top-level fields
    for field in REQUIRED_TOP_LEVEL:
        if field not in manifest:
            result.error(f"Missing required top-level field: '{field}'")

    # Version check
    version = manifest.get("version")
    if version is not None and version != RUNBUNDLE_VERSION:
        result.error(
            f"Invalid version: '{version}' (expected '{RUNBUNDLE_VERSION}')"
        )

    # spec_id
    spec_id = manifest.get("spec_id")
    if spec_id is not None and (not isinstance(spec_id, str) or len(spec_id) == 0):
        result.error("'spec_id' must be a non-empty string")

    # timestamp
    ts = manifest.get("timestamp")
    if ts is not None:
        if not isinstance(ts, str) or _parse_iso8601(ts) is None:
            result.error(f"'timestamp' is not valid ISO 8601: '{ts}'")

    # provenance
    prov = manifest.get("provenance")
    if isinstance(prov, dict):
        for field in REQUIRED_PROVENANCE:
            if field not in prov:
                result.error(f"Missing provenance field: '{field}'")
        # seeds must be non-empty list
        seeds = prov.get("seeds")
        if seeds is not None:
            if not isinstance(seeds, list) or len(seeds) == 0:
                result.error("'provenance.seeds' must be a non-empty list")
        # git_hash length
        gh = prov.get("git_hash")
        if gh is not None and (not isinstance(gh, str) or len(gh) < 7):
            result.error("'provenance.git_hash' must be at least 7 characters")
    elif prov is not None:
        result.error("'provenance' must be an object")

    # metrics
    metrics = manifest.get("metrics")
    if isinstance(metrics, dict):
        for field in REQUIRED_METRICS:
            if field not in metrics:
                result.error(f"Missing required metric: '{field}'")
            elif not _is_finite_float(metrics[field]):
                result.error(
                    f"Metric '{field}' must be a finite float, got: {metrics[field]}"
                )
    elif metrics is not None:
        result.error("'metrics' must be an object")

    # artifacts keys
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        for key in REQUIRED_ARTIFACTS:
            if key not in artifacts:
                result.error(f"Missing required artifact: '{key}'")
    elif artifacts is not None:
        result.error("'artifacts' must be an object")

    # hashes must cover all artifacts
    hashes = manifest.get("hashes")
    if isinstance(artifacts, dict) and isinstance(hashes, dict):
        for key in artifacts:
            if key not in hashes:
                result.error(
                    f"Artifact '{key}' has no corresponding entry in 'hashes'"
                )
            else:
                h = hashes[key]
                if not isinstance(h, str) or not h.startswith("sha256:"):
                    result.error(
                        f"Hash for '{key}' must start with 'sha256:', got: '{h}'"
                    )


def validate_artifact_files(
    manifest: dict, bundle_dir: Path, result: ValidationResult
) -> None:
    """Check that artifact files exist and hashes match."""
    artifacts = manifest.get("artifacts", {})
    hashes = manifest.get("hashes", {})

    for key, rel_path in artifacts.items():
        full_path = bundle_dir / rel_path
        if not full_path.is_file():
            result.error(f"Artifact file not found: '{rel_path}' (key: '{key}')")
            continue

        # Verify hash
        expected_hash = hashes.get(key, "")
        if expected_hash.startswith("sha256:"):
            expected_hex = expected_hash[len("sha256:"):]
            actual_hex = sha256_of_file(full_path)
            if actual_hex != expected_hex:
                result.error(
                    f"Hash mismatch for '{key}': "
                    f"expected {expected_hex[:16]}..., "
                    f"got {actual_hex[:16]}..."
                )


def validate_runbundle(bundle_path: Path, verbose: bool = False) -> ValidationResult:
    """Validate a RunBundle directory.

    Args:
        bundle_path: Path to the RunBundle directory (already extracted).
        verbose: If True, print detailed progress.

    Returns:
        ValidationResult with errors and warnings.
    """
    result = ValidationResult()

    # Find manifest
    manifest_path = bundle_path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        # Try one level down (zip may contain a single top-level folder)
        subdirs = [d for d in bundle_path.iterdir() if d.is_dir()]
        for sd in subdirs:
            candidate = sd / MANIFEST_FILENAME
            if candidate.is_file():
                manifest_path = candidate
                bundle_path = sd
                break
        else:
            result.error(f"'{MANIFEST_FILENAME}' not found in bundle")
            return result

    if verbose:
        print(f"Found manifest: {manifest_path}")

    # Parse JSON
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        result.error(f"Invalid JSON in manifest: {e}")
        return result

    # Validate structure
    validate_manifest(manifest, result)

    # Validate artifact files and hashes
    if manifest.get("artifacts") and manifest.get("hashes"):
        validate_artifact_files(manifest, bundle_path, result)

    return result


def validate_zip(zip_path: Path, verbose: bool = False) -> ValidationResult:
    """Validate a RunBundle submission zip file.

    Extracts to a temp directory, validates, then cleans up.
    """
    result = ValidationResult()

    if not zip_path.is_file():
        result.error(f"File not found: {zip_path}")
        return result

    if not zipfile.is_zipfile(str(zip_path)):
        result.error(f"Not a valid zip file: {zip_path}")
        return result

    tmpdir = tempfile.mkdtemp(prefix="pwm_validate_")
    try:
        if verbose:
            print(f"Extracting to {tmpdir}...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(tmpdir)
        result = validate_runbundle(Path(tmpdir), verbose=verbose)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate PWM RunBundle submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python community/validate.py submission.zip\n"
            "  python community/validate.py ./my_runbundle/\n"
        ),
    )
    parser.add_argument(
        "submission",
        type=str,
        help="Path to submission zip file or RunBundle directory",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed progress"
    )
    args = parser.parse_args()

    submission = Path(args.submission)

    if submission.is_dir():
        result = validate_runbundle(submission, verbose=args.verbose)
    elif submission.suffix == ".zip":
        result = validate_zip(submission, verbose=args.verbose)
    else:
        print(f"ERROR: '{submission}' is not a .zip file or directory")
        sys.exit(1)

    print(result.summary())
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
