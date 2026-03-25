"""pwm_core.cli.install

``pwm install`` — Plugin marketplace client for PWM.

Usage
-----
    pwm install gap_tv_enhanced           # install latest
    pwm install gap_tv_enhanced==1.2.0    # install specific version
    pwm list-plugins                      # list available plugins
    pwm uninstall gap_tv_enhanced         # remove plugin

Plugins are installed to ~/.pwm/plugins/<plugin_id>/
Registry is fetched from the contrib/plugin_registry/registry.yaml
(local first, then remote if --remote flag given).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # PyYAML — already in pwm_core deps

logger = logging.getLogger(__name__)

# Local registry path (relative to this file's package root)
_LOCAL_REGISTRY = Path(__file__).parent.parent.parent.parent / "contrib" / "plugin_registry" / "registry.yaml"
_PLUGINS_DIR = Path.home() / ".pwm" / "plugins"
_INSTALL_MANIFEST = Path.home() / ".pwm" / "installed_plugins.json"

# Remote registry URL (fallback)
_REMOTE_REGISTRY_URL = "https://raw.githubusercontent.com/integritynoble/Physics_World_Model/master/contrib/plugin_registry/registry.yaml"


def add_install_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("install", help="Install a PWM plugin from the marketplace")
    p.add_argument("plugin_spec", help="Plugin ID (e.g. gap_tv_enhanced or gap_tv_enhanced==1.2.0)")
    p.add_argument("--remote", action="store_true", help="Fetch registry from remote GitHub (default: local)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be installed without installing")
    p.set_defaults(func=cmd_install)

    p_list = subparsers.add_parser("list-plugins", help="List available plugins in the registry")
    p_list.add_argument("--modality", default="", help="Filter by modality")
    p_list.add_argument("--remote", action="store_true")
    p_list.set_defaults(func=cmd_list_plugins)

    p_un = subparsers.add_parser("uninstall", help="Remove an installed PWM plugin")
    p_un.add_argument("plugin_id", help="Plugin ID to remove")
    p_un.set_defaults(func=cmd_uninstall)


def cmd_install(args: argparse.Namespace) -> int:
    spec = args.plugin_spec
    plugin_id, version_req = _parse_plugin_spec(spec)
    registry = _load_registry(remote=args.remote)
    entry = _find_plugin(registry, plugin_id, version_req)
    if entry is None:
        print(f"[pwm install] ERROR: plugin '{plugin_id}' not found in registry.")
        _suggest_similar(registry, plugin_id)
        return 1

    print(f"[pwm install] Found: {entry['display_name']} v{entry['version']}")
    print(f"  author:     {entry['author']}")
    print(f"  license:    {entry['license']}")
    print(f"  modalities: {', '.join(entry['modalities'])}")
    print(f"  trust_score: {entry['trust_score']} / 5.0  ({entry['certified_runs']} certified runs)")

    if args.dry_run:
        print(f"[pwm install] Dry run — not installing.")
        return 0

    install_dir = _PLUGINS_DIR / entry["id"]
    install_dir.mkdir(parents=True, exist_ok=True)

    # Write plugin metadata
    meta_path = install_dir / "plugin.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)

    # Update installed manifest
    manifest = _load_manifest()
    manifest[entry["id"]] = {
        "version": entry["version"],
        "install_dir": str(install_dir),
        "entry_point": entry["entry_point"],
        "modalities": entry["modalities"],
    }
    _save_manifest(manifest)

    print(f"[pwm install] Installed {entry['id']} v{entry['version']} → {install_dir}")
    print(f"  To use: from pwm_core.plugins import load_plugin; load_plugin('{entry['id']}')")
    return 0


def cmd_list_plugins(args: argparse.Namespace) -> int:
    registry = _load_registry(remote=args.remote)
    plugins = registry.get("plugins", [])
    if args.modality:
        plugins = [p for p in plugins if args.modality in p.get("modalities", [])]

    installed = _load_manifest()

    print(f"\n{'ID':<28} {'Version':<10} {'Trust':>6} {'Certified':>10}  Modalities")
    print("-" * 80)
    for p in plugins:
        is_installed = " [installed]" if p["id"] in installed else ""
        mods = ", ".join(p["modalities"][:3])
        if len(p["modalities"]) > 3:
            mods += f" +{len(p['modalities'])-3}"
        print(f"{p['id']:<28} {p['version']:<10} {p['trust_score']:>6.1f} {p['certified_runs']:>10}  {mods}{is_installed}")
    print(f"\n{len(plugins)} plugin(s) available. Use `pwm install <id>` to install.\n")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    manifest = _load_manifest()
    if args.plugin_id not in manifest:
        print(f"[pwm uninstall] Plugin '{args.plugin_id}' is not installed.")
        return 1
    install_dir = Path(manifest[args.plugin_id]["install_dir"])
    if install_dir.exists():
        shutil.rmtree(install_dir)
    del manifest[args.plugin_id]
    _save_manifest(manifest)
    print(f"[pwm uninstall] Removed {args.plugin_id}.")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_plugin_spec(spec: str):
    if "==" in spec:
        parts = spec.split("==", 1)
        return parts[0].strip(), parts[1].strip()
    return spec.strip(), None


def _load_registry(remote: bool = False) -> Dict[str, Any]:
    if not remote and _LOCAL_REGISTRY.exists():
        with open(_LOCAL_REGISTRY, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    # Fall back to remote
    try:
        with urllib.request.urlopen(_REMOTE_REGISTRY_URL, timeout=10) as resp:
            content = resp.read().decode("utf-8")
        return yaml.safe_load(content) or {}
    except Exception as exc:
        logger.warning("Could not fetch remote registry: %s", exc)
        if _LOCAL_REGISTRY.exists():
            with open(_LOCAL_REGISTRY, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {"plugins": []}


def _find_plugin(registry: Dict, plugin_id: str, version: Optional[str]) -> Optional[Dict]:
    for p in registry.get("plugins", []):
        if p["id"] == plugin_id:
            if version is None or p["version"] == version:
                return p
    return None


def _suggest_similar(registry: Dict, plugin_id: str) -> None:
    candidates = [p["id"] for p in registry.get("plugins", []) if plugin_id[:4] in p["id"]]
    if candidates:
        print(f"  Did you mean: {', '.join(candidates[:3])}?")


def _load_manifest() -> Dict[str, Any]:
    if _INSTALL_MANIFEST.exists():
        with open(_INSTALL_MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: Dict[str, Any]) -> None:
    _INSTALL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(_INSTALL_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
