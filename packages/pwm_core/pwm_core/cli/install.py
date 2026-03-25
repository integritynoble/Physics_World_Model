"""pwm_core.cli.install

``pwm install`` — install a community solver plugin.

Usage
-----
    pwm install gap_tv                       # install from built-in registry
    pwm install ./my_solver/                 # install from local directory
    pwm install https://github.com/user/repo # install from GitHub

Plugins are validated against the MethodCard schema before installation.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def add_install_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "install",
        help="Install a community solver plugin",
    )
    p.add_argument("plugin", help="Plugin name, local path, or GitHub URL")
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip MethodCard schema validation (not recommended)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing plugin with the same name",
    )
    p.set_defaults(func=cmd_install)


def cmd_install(args: argparse.Namespace) -> int:
    plugin_ref = args.plugin
    validate = not args.no_validate

    print(f"[pwm install] Installing plugin: {plugin_ref}")

    # Determine source type
    if plugin_ref.startswith("https://") or plugin_ref.startswith("git@"):
        return _install_from_git(plugin_ref, validate=validate)
    elif Path(plugin_ref).exists():
        return _install_from_local(Path(plugin_ref), validate=validate)
    else:
        return _install_from_registry(plugin_ref, validate=validate)


def _install_from_registry(name: str, validate: bool) -> int:
    """Install a named plugin from the built-in contrib registry."""
    try:
        from pwm_core.targeting.plugin_loader import install_plugin
        install_plugin(name, validate=validate)
        print(f"[pwm install] Plugin '{name}' installed successfully.")
        return 0
    except ImportError:
        logger.warning("plugin_loader not available")
    except Exception as exc:
        print(f"[pwm install] Error: {exc}")
        return 1

    print(f"[pwm install] Plugin '{name}' not found in registry.")
    print("  Run 'pwm plugins list' to see available plugins.")
    return 1


def _install_from_local(path: Path, validate: bool) -> int:
    """Install a plugin from a local directory."""
    if not path.exists():
        print(f"[pwm install] Error: path does not exist: {path}")
        return 1

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", str(path)],
            check=True,
        )
        print(f"[pwm install] Installed from {path}")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"[pwm install] pip install failed: {exc}")
        return 1


def _install_from_git(url: str, validate: bool) -> int:
    """Install a plugin from a git repository."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", f"git+{url}"],
            check=True,
        )
        print(f"[pwm install] Installed from {url}")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"[pwm install] pip install failed: {exc}")
        return 1
