"""pwm_core.core.runbundle.provenance

Expanded provenance capture for reproducibility (Plan v3, Section 18).

Captures:
- Git state (hash, dirty flag)
- Platform (OS, architecture, hostname)
- Python + key package versions
- Random seeds used
- SHA256 hashes of input arrays and agent reports
- ISO-8601 timestamp
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional

import numpy as np


def _git_hash() -> Optional[str]:
    """Get current git commit hash, or None if not in a repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    """Check if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        return len(result.stdout.strip()) > 0
    except Exception:
        return None


def _package_version(name: str) -> Optional[str]:
    """Get installed package version, or None."""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _hash_dict(d: Dict[str, Any]) -> str:
    """Compute SHA256 of a JSON-serialized dict (deterministic key order)."""
    payload = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_array(arr: np.ndarray) -> str:
    """Compute SHA256 of a numpy array's raw bytes."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def capture_provenance(
    rb_dir: str,
    seeds: Optional[List[int]] = None,
    array_hashes: Optional[Dict[str, str]] = None,
    agent_reports: Optional[Dict[str, Any]] = None,
    spec_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """Capture expanded provenance and write to ``provenance.json``.

    Parameters
    ----------
    rb_dir : str
        RunBundle directory where ``provenance.json`` will be written.
    seeds : list[int], optional
        Random seeds used during the run.
    array_hashes : dict[str, str], optional
        Pre-computed SHA256 hashes of key arrays (e.g. ``{"y": "abc...", "x_true": "def..."}``).
    agent_reports : dict[str, Any], optional
        Serialized agent reports for hashing.
    spec_dict : dict[str, Any], optional
        Serialized ExperimentSpec for hashing.

    Returns
    -------
    str
        Path to the written ``provenance.json`` file.
    """
    prov: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # Git state
        "git_hash": _git_hash(),
        "git_dirty": _git_dirty(),
        # Platform
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
            "platform_string": platform.platform(),
        },
        # Python
        "python_version": sys.version,
        # Package versions
        "packages": {
            "numpy": _package_version("numpy"),
            "pydantic": _package_version("pydantic"),
            "pwm_core": _package_version("pwm-core"),
        },
        # Seeds
        "seeds": seeds or [],
        # Array hashes
        "array_hashes": array_hashes or {},
        # Agent reports hash (hash of all reports together)
        "agent_reports_hash": (
            _hash_dict(agent_reports) if agent_reports else None
        ),
        # Spec hash
        "spec_hash": _hash_dict(spec_dict) if spec_dict else None,
    }

    path = os.path.join(rb_dir, "provenance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    return path
