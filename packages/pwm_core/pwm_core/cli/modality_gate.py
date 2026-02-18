"""pwm_core.cli.modality_gate
==============================

One-at-a-time modality execution guardrail.

Provides the ``pwm next-modality`` CLI subcommand that checks whether the
previous modality is fully DONE before revealing the next one in the
execution order.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Canonical execution order (64 modalities)
# ---------------------------------------------------------------------------

EXECUTION_ORDER = [
    # Tier 1 - Core compressive
    "spc", "cassi", "cacti", "ct", "mri",
    # Tier 2 - Microscopy fundamentals
    "widefield", "widefield_lowdose", "confocal_livecell", "confocal_3d",
    "sim", "lensless", "lightsheet", "flim",
    # Tier 3 - Coherent imaging
    "ptychography", "holography", "phase_retrieval", "fpm", "oct",
    # Tier 4 - Medical imaging
    "xray_radiography", "ultrasound", "photoacoustic", "dot",
    "pet", "spect", "fluoroscopy", "mammography", "dexa", "cbct",
    # Tier 5 - Neural rendering + computational
    "nerf", "gaussian_splatting", "matrix", "panorama", "light_field", "integral",
    # Tier 6 - Electron microscopy
    "sem", "tem", "stem", "electron_tomography",
    "electron_diffraction", "ebsd", "eels",
    # Tier 7 - Advanced medical
    "angiography", "doppler_ultrasound", "elastography", "fmri", "mrs", "diffusion_mri",
    # Tier 8 - Advanced microscopy
    "two_photon", "sted", "palm_storm", "tirf", "polarization",
    # Tier 9 - Clinical optics + depth
    "endoscopy", "fundus", "octa", "tof_camera", "lidar", "structured_light",
    # Tier 10 - Remote sensing + exotic
    "sar", "sonar", "electron_holography", "neutron_tomo", "proton_radiography", "muon_tomo",
]

# Project root -- assumes CLI is run from the repo root or a child directory.
_PROJECT_ROOT = Path.cwd()


def _scoreboard_path() -> Path:
    """Return the path to the scoreboard YAML file."""
    return _PROJECT_ROOT / "pwm" / "reports" / "scoreboard.yaml"


def _load_scoreboard() -> dict:
    """Load the scoreboard YAML, returning an empty dict on failure."""
    sb_path = _scoreboard_path()
    if not sb_path.exists():
        return {}
    if yaml is None:
        return {}
    with open(sb_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def check_modality_done(modality_key: str) -> Tuple[bool, List[str]]:
    """Check whether a modality is fully DONE.

    Criteria
    --------
    1. Report file exists at ``pwm/reports/{modality_key}.md``
    2. Scoreboard entry has status ``DONE``
    3. RunBundle path listed in the scoreboard exists on disk

    Parameters
    ----------
    modality_key:
        The canonical modality string (e.g. ``"cassi"``).

    Returns
    -------
    (all_pass, failures):
        *all_pass* is True when every check passes.
        *failures* is a list of human-readable failure descriptions.
    """
    failures: List[str] = []

    # 1. Report file
    report = _PROJECT_ROOT / "pwm" / "reports" / f"{modality_key}.md"
    if not report.exists():
        failures.append(f"Report not found: {report}")

    # 2. Scoreboard status
    sb = _load_scoreboard()
    entry = sb.get(modality_key)
    if entry is None:
        failures.append(f"Scoreboard has no entry for '{modality_key}'")
    else:
        status = entry.get("status", "").upper() if isinstance(entry, dict) else str(entry).upper()
        if status != "DONE":
            failures.append(f"Scoreboard status for '{modality_key}' is '{status}', expected 'DONE'")

        # 3. RunBundle path
        if isinstance(entry, dict):
            bundle_path = entry.get("runbundle")
            if bundle_path:
                if not Path(bundle_path).exists():
                    failures.append(f"RunBundle path does not exist: {bundle_path}")
            else:
                failures.append(f"Scoreboard entry for '{modality_key}' has no runbundle path")

    all_pass = len(failures) == 0
    return all_pass, failures


def get_next_modality() -> Optional[str]:
    """Return the first modality in *EXECUTION_ORDER* whose status is not DONE.

    Returns ``None`` when every modality is DONE.
    """
    sb = _load_scoreboard()
    for key in EXECUTION_ORDER:
        entry = sb.get(key)
        if entry is None:
            return key
        status = entry.get("status", "").upper() if isinstance(entry, dict) else str(entry).upper()
        if status != "DONE":
            return key
    return None


def cmd_next_modality(args):
    """CLI handler for ``pwm next-modality``."""
    next_key = get_next_modality()

    if next_key is None:
        print("All 64 modalities are DONE.")
        return

    idx = EXECUTION_ORDER.index(next_key)

    # If this is the very first modality, no predecessor to check
    if idx == 0:
        print(f"Next modality: {next_key}  (first in execution order)")
        return

    prev_key = EXECUTION_ORDER[idx - 1]
    done, failures = check_modality_done(prev_key)

    if done:
        print(f"Previous modality '{prev_key}' is DONE. Gate passed.")
        print(f"Next modality: {next_key}")
    else:
        print(f"BLOCKED: previous modality '{prev_key}' is NOT done.")
        for f in failures:
            print(f"  - {f}")
        print(f"Finish '{prev_key}' before starting '{next_key}'.")
        sys.exit(1)


def add_next_modality_subparser(subparsers):
    """Register the ``next-modality`` subcommand."""
    p = subparsers.add_parser("next-modality", help="Check gate and show next modality")
    p.set_defaults(func=cmd_next_modality)
