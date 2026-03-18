"""pwm_core.io.manifest

Dataset manifest validation.

A manifest is a JSON dict with required fields for dataset provenance,
licensing, and sample inventory. This module validates manifest structure
and can batch-validate all manifests in a directory.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


REQUIRED_FIELDS = {"dataset_id", "modality", "samples", "license"}
SAMPLE_FIELDS = {"id", "path", "split"}


def validate_manifest(manifest: dict) -> List[str]:
    """Validate a dataset manifest dict.

    Returns list of error strings (empty = valid).
    """
    errors: List[str] = []

    # Check required top-level fields
    for field in REQUIRED_FIELDS:
        if field not in manifest:
            errors.append(f"Missing required field: '{field}'")

    # Validate dataset_id
    if "dataset_id" in manifest:
        did = manifest["dataset_id"]
        if not isinstance(did, str) or not did.strip():
            errors.append("'dataset_id' must be a non-empty string")

    # Validate modality
    if "modality" in manifest:
        mod = manifest["modality"]
        if not isinstance(mod, str) or not mod.strip():
            errors.append("'modality' must be a non-empty string")

    # Validate license
    if "license" in manifest:
        lic = manifest["license"]
        if not isinstance(lic, str) or not lic.strip():
            errors.append("'license' must be a non-empty string")

    # Validate samples
    if "samples" in manifest:
        samples = manifest["samples"]
        if not isinstance(samples, list):
            errors.append("'samples' must be a list")
        else:
            for i, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    errors.append(f"samples[{i}]: must be a dict")
                    continue
                for sf in SAMPLE_FIELDS:
                    if sf not in sample:
                        errors.append(f"samples[{i}]: missing field '{sf}'")
                # Validate sample id
                if "id" in sample and not isinstance(sample["id"], str):
                    errors.append(f"samples[{i}]: 'id' must be a string")
                # Validate split
                if "split" in sample:
                    valid_splits = {"train", "val", "test"}
                    if sample["split"] not in valid_splits:
                        errors.append(
                            f"samples[{i}]: 'split' must be one of {valid_splits}, "
                            f"got '{sample['split']}'"
                        )

    return errors


def validate_manifest_file(path: str) -> List[str]:
    """Load JSON manifest from path and validate."""
    errors: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return [f"File not found: {path}"]
    except Exception as e:
        return [f"Error reading file: {e}"]

    return validate_manifest(manifest)


def validate_all_manifests(manifests_dir: str) -> Dict[str, List[str]]:
    """Validate all .json manifests in a directory.

    Returns dict mapping filename -> list of errors (empty list = valid).
    """
    results: Dict[str, List[str]] = {}

    if not os.path.isdir(manifests_dir):
        return {"_error": [f"Directory not found: {manifests_dir}"]}

    for fname in sorted(os.listdir(manifests_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(manifests_dir, fname)
            results[fname] = validate_manifest_file(fpath)

    return results
