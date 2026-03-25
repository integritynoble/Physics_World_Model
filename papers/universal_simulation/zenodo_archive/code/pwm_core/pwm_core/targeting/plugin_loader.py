"""pwm_core.targeting.plugin_loader
====================================

External plugin loading and tier separation for PWM contributions.

``pwm install`` installs solvers/calibrators from git repos or local dirs
without requiring a fork-and-PR workflow.

Plugin Tiers (per plan Addition 21):

| Tier      | Location                      | Leaderboard? | How to get there         |
|-----------|-------------------------------|-------------|--------------------------|
| Local     | ``~/.pwm/plugins/``           | No          | ``pwm install``          |
| Community | ``contrib/solvers/``          | Yes         | PR → fast-lane merge     |
| Official  | ``baselines/``                | Yes + badge | PWM team curates; frozen |

Usage::

    pwm install https://github.com/alice/my_solver
    pwm install ./my_local_solver
    pwm install ./my_solver --tier community
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import inspect
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CONTRIB_DIR = Path(__file__).resolve().parents[2] / "contrib"
_USER_PLUGIN_DIR = Path.home() / ".pwm" / "plugins"

# Plugin tiers
TIER_LOCAL = "local"
TIER_COMMUNITY = "community"
TIER_OFFICIAL = "official"

VALID_TIERS = {TIER_LOCAL, TIER_COMMUNITY, TIER_OFFICIAL}

# Required files in a plugin
REQUIRED_SOLVER_FILES = ["solver.py", "config.yaml"]
REQUIRED_CALIBRATOR_FILES = ["calibrator.py", "config.yaml"]

# Forbidden imports (same as contrib_check)
FORBIDDEN_IMPORTS = [
    "graph.compiler",
    "graph.primitives",
    "targeting",
    "pwm_core.graph.compiler",
    "pwm_core.graph.primitives",
    "pwm_core.targeting",
]


# ---------------------------------------------------------------------------
# Cryptographic plugin signing (HMAC-SHA256)
# ---------------------------------------------------------------------------


class PluginSignature:
    """HMAC-SHA256 signature for plugin authenticity verification."""

    ALGORITHM = "hmac-sha256"

    @staticmethod
    def compute_hash(plugin_dir: Path) -> str:
        """Compute a deterministic hash of plugin contents.

        Hashes all ``*.py`` files in sorted order plus ``plugin.yaml``
        (if present) to produce a single SHA-256 digest.
        """
        h = hashlib.sha256()
        # Hash all .py files in sorted order for determinism
        py_files = sorted(plugin_dir.rglob("*.py"))
        for f in py_files:
            h.update(f.read_bytes())
        # Hash config if present
        config = plugin_dir / "plugin.yaml"
        if config.exists():
            h.update(config.read_bytes())
        return h.hexdigest()

    @staticmethod
    def sign(plugin_dir: Path, secret_key: str) -> dict:
        """Sign a plugin directory.

        Returns signature dict and writes ``plugin.sig.json`` into
        *plugin_dir*.
        """
        content_hash = PluginSignature.compute_hash(plugin_dir)
        signature = hmac.new(
            secret_key.encode(), content_hash.encode(), hashlib.sha256
        ).hexdigest()

        import datetime

        sig_data = {
            "algorithm": PluginSignature.ALGORITHM,
            "content_hash": content_hash,
            "signature": signature,
            "signed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "signer": "pwm-plugin-authority",
        }

        sig_path = plugin_dir / "plugin.sig.json"
        sig_path.write_text(json.dumps(sig_data, indent=2))
        return sig_data

    @staticmethod
    def verify(plugin_dir: Path, secret_key: str) -> tuple[bool, str]:
        """Verify a plugin's cryptographic signature.

        Returns
        -------
        tuple of (bool, str)
            ``(valid, message)`` where *valid* is True when the HMAC
            matches and the content hash is unchanged.
        """
        sig_path = plugin_dir / "plugin.sig.json"
        if not sig_path.exists():
            return False, "No plugin.sig.json found — plugin is unsigned"

        try:
            sig_data = json.loads(sig_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            return False, f"Cannot read plugin.sig.json: {e}"

        if sig_data.get("algorithm") != PluginSignature.ALGORITHM:
            return False, (
                f"Unknown signature algorithm: {sig_data.get('algorithm')}"
            )

        # Recompute content hash
        current_hash = PluginSignature.compute_hash(plugin_dir)
        if current_hash != sig_data.get("content_hash"):
            return False, (
                "Content hash mismatch — plugin files were modified after signing"
            )

        # Verify HMAC
        expected_sig = hmac.new(
            secret_key.encode(), current_hash.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected_sig, sig_data.get("signature", "")):
            return False, (
                "HMAC signature verification failed — "
                "invalid key or tampered signature"
            )

        return True, (
            f"Plugin signature valid (signed {sig_data.get('signed_at', 'unknown')})"
        )


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------


def get_plugin_dir(tier: str) -> Path:
    """Return the directory for a given plugin tier."""
    if tier == TIER_LOCAL:
        return _USER_PLUGIN_DIR
    elif tier == TIER_COMMUNITY:
        return _CONTRIB_DIR / "solvers"
    elif tier == TIER_OFFICIAL:
        return _CONTRIB_DIR.parent / "baselines"
    else:
        raise ValueError(f"Unknown tier: {tier}. Must be one of {VALID_TIERS}")


def list_plugins(tier: Optional[str] = None) -> List[Dict[str, Any]]:
    """List installed plugins, optionally filtered by tier.

    Returns
    -------
    list of dict
        Each dict has: name, tier, path, has_config.
    """
    plugins = []

    tiers = [tier] if tier else [TIER_LOCAL, TIER_COMMUNITY, TIER_OFFICIAL]

    for t in tiers:
        plugin_dir = get_plugin_dir(t)
        if not plugin_dir.is_dir():
            continue
        for child in sorted(plugin_dir.iterdir()):
            if child.is_dir() and (child / "solver.py").exists():
                plugins.append({
                    "name": child.name,
                    "tier": t,
                    "path": str(child),
                    "has_config": (child / "config.yaml").exists(),
                })

    return plugins


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_plugin_structure(
    plugin_dir: Path, plugin_type: str = "solver"
) -> List[str]:
    """Validate that a plugin directory has required files.

    Returns list of error messages (empty = valid).
    """
    errors = []
    required = (
        REQUIRED_CALIBRATOR_FILES if plugin_type == "calibrator"
        else REQUIRED_SOLVER_FILES
    )

    for fname in required:
        if not (plugin_dir / fname).exists():
            errors.append(f"Missing required file: {fname}")

    return errors


def _validate_solver_signature(solver_path: Path) -> List[str]:
    """Check that the solver has the correct function signature."""
    errors = []
    try:
        spec = importlib.util.spec_from_file_location("_plugin_solver", solver_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        return [f"Cannot load solver module: {e}"]

    # Find the run_* function
    run_fns = [
        name for name in dir(mod)
        if name.startswith("run_") and callable(getattr(mod, name))
    ]

    if not run_fns:
        errors.append("No run_<name>() function found in solver.py")
        return errors

    for fn_name in run_fns:
        fn = getattr(mod, fn_name)
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        if params != ["y", "physics", "cfg"]:
            errors.append(
                f"{fn_name} has wrong signature: ({', '.join(params)}) "
                f"— expected (y, physics, cfg)"
            )

    return errors


def _validate_isolation(source_path: Path) -> List[str]:
    """Check that source code doesn't import forbidden modules."""
    errors = []
    try:
        source = source_path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"Cannot read source: {e}"]

    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for pattern in FORBIDDEN_IMPORTS:
            if pattern in stripped:
                errors.append(f"Forbidden import '{pattern}' in: {stripped}")

    return errors


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load config.yaml from a plugin directory."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def validate_plugin(
    plugin_dir: Path, plugin_type: str = "solver"
) -> Dict[str, Any]:
    """Full validation of a plugin directory.

    Returns
    -------
    dict
        Keys: valid (bool), errors (list of str), config (dict or None).
    """
    errors = []

    # 1. Structure
    errors.extend(_validate_plugin_structure(plugin_dir, plugin_type))
    if errors:
        return {"valid": False, "errors": errors, "config": None}

    # 2. Config
    config_path = plugin_dir / "config.yaml"
    try:
        config = _load_config(config_path)
    except Exception as e:
        errors.append(f"Invalid config.yaml: {e}")
        config = None

    if config and not config.get("name"):
        errors.append("config.yaml must contain 'name' field")

    # 3. Source validation
    if plugin_type == "solver":
        solver_path = plugin_dir / "solver.py"
        errors.extend(_validate_solver_signature(solver_path))
        errors.extend(_validate_isolation(solver_path))
    elif plugin_type == "calibrator":
        cal_path = plugin_dir / "calibrator.py"
        errors.extend(_validate_isolation(cal_path))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def _clone_repo(url: str, dest: Path) -> Path:
    """Clone a git repository to a temporary directory."""
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"git clone failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("git is not installed or not in PATH")
    return dest


def install_plugin(
    source: str,
    plugin_type: str = "solver",
    tier: str = TIER_LOCAL,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Install a plugin from a git URL or local directory.

    Parameters
    ----------
    source : str
        Git URL or local directory path.
    plugin_type : str
        'solver' or 'calibrator'.
    tier : str
        Target tier: 'local' (default) or 'community'.
        'official' tier requires manual curation.
    name : str, optional
        Override plugin name (default: directory name).

    Returns
    -------
    dict
        Keys: installed (bool), name (str), tier (str), path (str),
        errors (list of str).
    """
    if tier == TIER_OFFICIAL:
        return {
            "installed": False,
            "name": name or "?",
            "tier": tier,
            "path": "",
            "errors": ["Cannot install directly to official tier. "
                       "Official baselines require governance vote."],
        }

    if tier not in {TIER_LOCAL, TIER_COMMUNITY}:
        return {
            "installed": False,
            "name": name or "?",
            "tier": tier,
            "path": "",
            "errors": [f"Invalid tier: {tier}"],
        }

    # Resolve source
    source_path = Path(source)
    cleanup_tmp = False

    if source_path.is_dir():
        plugin_dir = source_path
    elif source.startswith("http") or source.startswith("git@"):
        # Git clone to temp dir
        tmp = Path(tempfile.mkdtemp())
        cleanup_tmp = True
        try:
            plugin_dir = _clone_repo(source, tmp)
        except RuntimeError as e:
            return {
                "installed": False,
                "name": name or "?",
                "tier": tier,
                "path": "",
                "errors": [str(e)],
            }
    else:
        return {
            "installed": False,
            "name": name or source,
            "tier": tier,
            "path": "",
            "errors": [f"Source not found: {source}"],
        }

    try:
        # Validate
        result = validate_plugin(plugin_dir, plugin_type)
        if not result["valid"]:
            return {
                "installed": False,
                "name": name or plugin_dir.name,
                "tier": tier,
                "path": str(plugin_dir),
                "errors": result["errors"],
            }

        # Determine name
        plugin_name = name or (result["config"] or {}).get("name") or plugin_dir.name

        # Determine destination
        target_root = get_plugin_dir(tier)
        if tier == TIER_COMMUNITY:
            if plugin_type == "calibrator":
                target_root = _CONTRIB_DIR / "calibrators"
            else:
                target_root = _CONTRIB_DIR / "solvers"

        dest = target_root / plugin_name

        if dest.exists():
            return {
                "installed": False,
                "name": plugin_name,
                "tier": tier,
                "path": str(dest),
                "errors": [f"Plugin already exists at {dest}. "
                           "Remove it first or use a different name."],
            }

        # Copy
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(plugin_dir, dest)

        # Write tier metadata
        tier_meta = dest / ".pwm_tier"
        tier_meta.write_text(tier, encoding="utf-8")

        logger.info(f"Installed {plugin_type} '{plugin_name}' to {dest} (tier={tier})")

        return {
            "installed": True,
            "name": plugin_name,
            "tier": tier,
            "path": str(dest),
            "errors": [],
        }

    finally:
        if cleanup_tmp:
            shutil.rmtree(plugin_dir, ignore_errors=True)


def uninstall_plugin(name: str, tier: str = TIER_LOCAL) -> Dict[str, Any]:
    """Remove an installed plugin.

    Only local and community tier plugins can be uninstalled.
    Official baselines are frozen.
    """
    if tier == TIER_OFFICIAL:
        return {
            "removed": False,
            "errors": ["Cannot uninstall official baselines. They are frozen."],
        }

    plugin_dir = get_plugin_dir(tier) / name
    if not plugin_dir.exists():
        return {
            "removed": False,
            "errors": [f"Plugin '{name}' not found in {tier} tier"],
        }

    shutil.rmtree(plugin_dir)
    logger.info(f"Uninstalled plugin '{name}' from {tier} tier")

    return {"removed": True, "errors": []}


# ---------------------------------------------------------------------------
# Tier queries
# ---------------------------------------------------------------------------


def get_plugin_tier(name: str) -> Optional[str]:
    """Determine which tier a plugin belongs to."""
    for tier in [TIER_OFFICIAL, TIER_COMMUNITY, TIER_LOCAL]:
        plugin_dir = get_plugin_dir(tier) / name
        if plugin_dir.is_dir():
            return tier
    return None


def is_leaderboard_eligible(name: str) -> bool:
    """Check if a plugin's results can appear on the leaderboard.

    Only community and official tier plugins are eligible.
    """
    tier = get_plugin_tier(name)
    return tier in {TIER_COMMUNITY, TIER_OFFICIAL}


def is_baseline(name: str) -> bool:
    """Check if a plugin is an official frozen baseline."""
    return get_plugin_tier(name) == TIER_OFFICIAL


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def cmd_install(args) -> int:
    """CLI handler for ``pwm install``.

    Supports ``--verify-signature`` flag to require a valid HMAC-SHA256
    signature before installation proceeds.  The signing key is read
    from the ``PWM_PLUGIN_SIGNING_KEY`` environment variable.
    """
    import os

    # --- optional signature verification before install ----------------
    verify_sig = getattr(args, "verify_signature", False)
    if verify_sig:
        signing_key = os.environ.get("PWM_PLUGIN_SIGNING_KEY", "")
        if not signing_key:
            print("ERROR: --verify-signature requires "
                  "PWM_PLUGIN_SIGNING_KEY env var to be set")
            return 1

        source_path = Path(args.source)
        if source_path.is_dir():
            valid, msg = PluginSignature.verify(source_path, signing_key)
            print(f"Signature check: {msg}")
            if not valid:
                print("Aborting install — signature verification failed.")
                return 1
        else:
            print("WARNING: --verify-signature only works with local "
                  "directories; skipping for remote sources")

    # --- normal install flow -------------------------------------------
    result = install_plugin(
        source=args.source,
        plugin_type=getattr(args, "type", "solver"),
        tier=getattr(args, "tier", TIER_LOCAL),
        name=getattr(args, "name", None),
    )

    if result["installed"]:
        print(f"Installed '{result['name']}' to {result['tier']} tier")
        print(f"  Path: {result['path']}")
        print(f"  Leaderboard eligible: {is_leaderboard_eligible(result['name'])}")
        print(f"\nRun: pwm evaluate --sandbox --modality widefield "
              f"--solver {result['name']}")
        return 0
    else:
        print(f"Installation failed for '{result['name']}':")
        for err in result["errors"]:
            print(f"  - {err}")
        return 1


def cmd_uninstall(args) -> int:
    """CLI handler for ``pwm uninstall``."""
    result = uninstall_plugin(
        name=args.name,
        tier=getattr(args, "tier", TIER_LOCAL),
    )

    if result["removed"]:
        print(f"Uninstalled '{args.name}'")
        return 0
    else:
        for err in result["errors"]:
            print(f"  - {err}")
        return 1


def cmd_plugins(args) -> int:
    """CLI handler for ``pwm plugins`` (list installed plugins)."""
    tier = getattr(args, "tier", None)
    plugins = list_plugins(tier=tier)

    if not plugins:
        print("No plugins installed.")
        return 0

    print(f"{'Name':<25} {'Tier':<12} {'Config':<8} Path")
    print("-" * 80)
    for p in plugins:
        cfg = "yes" if p["has_config"] else "no"
        badge = " [BASELINE]" if p["tier"] == TIER_OFFICIAL else ""
        print(f"{p['name']:<25} {p['tier']:<12} {cfg:<8} {p['path']}{badge}")

    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # Test validation
    tmp = Path(tempfile.mkdtemp())
    try:
        # Create a minimal solver plugin
        solver_dir = tmp / "test_solver"
        solver_dir.mkdir()

        (solver_dir / "solver.py").write_text(
            "import numpy as np\n\n"
            "def run_test_solver(y, physics, cfg):\n"
            "    return physics.adjoint(y), {'solver': 'test'}\n",
            encoding="utf-8",
        )
        (solver_dir / "config.yaml").write_text(
            "name: test_solver\nversion: 0.1.0\nfamily: classical\n",
            encoding="utf-8",
        )

        # Validate
        result = validate_plugin(solver_dir)
        assert result["valid"], f"Validation failed: {result['errors']}"

        # Test listing
        plugins = list_plugins()
        print(f"Found {len(plugins)} installed plugins")

        # Test tier detection
        assert get_plugin_tier("nonexistent") is None

        # Test cryptographic signing
        print("\nPlugin signing self-test:")
        sig = PluginSignature.sign(solver_dir, "self-test-key")
        print(f"  Signed: {sig['content_hash'][:16]}...")

        valid, msg = PluginSignature.verify(solver_dir, "self-test-key")
        print(f"  Verify (correct key): {valid} — {msg}")
        assert valid, f"Signature should be valid: {msg}"

        valid2, msg2 = PluginSignature.verify(solver_dir, "wrong-key")
        print(f"  Verify (wrong key):   {valid2} — {msg2}")
        assert not valid2, "Signature should fail with wrong key"

        # Tamper detection
        (solver_dir / "solver.py").write_text("# tampered\n", encoding="utf-8")
        valid3, msg3 = PluginSignature.verify(solver_dir, "self-test-key")
        print(f"  Verify (tampered):    {valid3} — {msg3}")
        assert not valid3, "Signature should fail after tampering"

        print("\nSelf-test PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
