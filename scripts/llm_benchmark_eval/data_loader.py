"""Load B1 JSON and B3 tar.gz benchmark data files."""

from __future__ import annotations

import json
import tarfile
from typing import Any

from .config import BENCHMARK_DATA_DIR


def load_b1_samples(variant: str) -> list[dict[str, Any]]:
    """Load the 50 B1 samples for a variant from its JSON file.

    Returns a list of sample dicts with keys:
        id, description, correct_spec, correct_dag, distractors
    """
    path = BENCHMARK_DATA_DIR / f"{variant}_b1_public.json"
    if not path.exists():
        raise FileNotFoundError(f"B1 data not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data["samples"]


def load_b3_samples(variant: str) -> list[dict[str, Any]]:
    """Load the 50 B3 samples for a variant from its tar.gz file.

    Extracts metadata.json from the archive (does not touch measurements.h5).

    Returns a list of sample dicts with keys:
        id, candidate_spec, label, true_spec
    """
    path = BENCHMARK_DATA_DIR / f"{variant}_b3_public.tar.gz"
    if not path.exists():
        raise FileNotFoundError(f"B3 data not found: {path}")
    with tarfile.open(path, "r:gz") as tf:
        member = tf.extractfile("metadata.json")
        if member is None:
            raise ValueError(f"metadata.json not found in {path}")
        data = json.load(member)
    return data["samples"]
