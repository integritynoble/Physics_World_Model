"""pwm_core.cli.doctor
======================

``pwm doctor`` — green/red environment checklist for friction-free first run.

Checks:
1. Python version >= 3.10
2. Core deps (numpy, scipy, pydantic, pyyaml, tqdm, rich, matplotlib)
3. Optional deps (torch, imageio, deepinv, streamlit)
4. YAML registries (6 files exist and parse)
5. Graph templates (26 templates compile)
6. CasePacks (at least 1 discoverable)
7. Disk/write (tmp dir writable)

Exit code 0 if all critical checks pass, 1 otherwise.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ANSI color codes with fallback
_SUPPORTS_COLOR: Optional[bool] = None


def _color_supported() -> bool:
    global _SUPPORTS_COLOR
    if _SUPPORTS_COLOR is not None:
        return _SUPPORTS_COLOR
    try:
        _SUPPORTS_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    except Exception:
        _SUPPORTS_COLOR = False
    return _SUPPORTS_COLOR


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m" if _color_supported() else text


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if _color_supported() else text


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m" if _color_supported() else text


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _color_supported() else text


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single doctor check."""
    name: str
    passed: bool
    critical: bool = True
    message: str = ""
    details: List[str] = field(default_factory=list)

    @property
    def status_str(self) -> str:
        if self.passed:
            return _green("PASS")
        elif self.critical:
            return _red("FAIL")
        else:
            return _yellow("WARN")


# ---------------------------------------------------------------------------
# Contrib directory
# ---------------------------------------------------------------------------

_CONTRIB_DIR = Path(__file__).resolve().parents[2] / "contrib"
_CASEPACKS_DIR = _CONTRIB_DIR / "casepacks"

# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

CORE_DEPS = ["numpy", "scipy", "pydantic", "yaml", "tqdm", "rich", "matplotlib"]
OPTIONAL_DEPS = ["torch", "imageio", "deepinv", "streamlit"]

REGISTRY_FILES = [
    "modalities.yaml",
    "solver_registry.yaml",
    "graph_templates.yaml",
    "primitives.yaml",
    "mismatch_db.yaml",
    "photon_db.yaml",
]


def check_python_version() -> CheckResult:
    """Check Python >= 3.10."""
    vi = sys.version_info
    ok = vi >= (3, 10)
    return CheckResult(
        name="Python version",
        passed=ok,
        critical=True,
        message=f"{vi.major}.{vi.minor}.{vi.micro}",
        details=[] if ok else ["Requires Python >= 3.10"],
    )


def check_core_deps() -> CheckResult:
    """Check core dependencies are importable."""
    missing = []
    for mod in CORE_DEPS:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    ok = len(missing) == 0
    return CheckResult(
        name="Core dependencies",
        passed=ok,
        critical=True,
        message=f"{len(CORE_DEPS) - len(missing)}/{len(CORE_DEPS)} available",
        details=[f"Missing: {', '.join(missing)}"] if missing else [],
    )


def check_optional_deps() -> CheckResult:
    """Check optional dependencies (non-critical)."""
    missing = []
    for mod in OPTIONAL_DEPS:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    ok = len(missing) == 0
    return CheckResult(
        name="Optional dependencies",
        passed=ok,
        critical=False,
        message=f"{len(OPTIONAL_DEPS) - len(missing)}/{len(OPTIONAL_DEPS)} available",
        details=[f"Not installed: {', '.join(missing)}"] if missing else [],
    )


def check_yaml_registries() -> CheckResult:
    """Check all 6 YAML registry files exist and parse."""
    import yaml as _yaml

    missing = []
    parse_errors = []
    for fname in REGISTRY_FILES:
        p = _CONTRIB_DIR / fname
        if not p.exists():
            missing.append(fname)
            continue
        try:
            with open(p) as f:
                _yaml.safe_load(f)
        except Exception as exc:
            parse_errors.append(f"{fname}: {exc}")

    ok = len(missing) == 0 and len(parse_errors) == 0
    details = []
    if missing:
        details.append(f"Missing: {', '.join(missing)}")
    if parse_errors:
        details.extend(parse_errors)
    return CheckResult(
        name="YAML registries",
        passed=ok,
        critical=True,
        message=f"{len(REGISTRY_FILES) - len(missing) - len(parse_errors)}/{len(REGISTRY_FILES)} OK",
        details=details,
    )


def check_graph_templates() -> CheckResult:
    """Check that all graph templates compile without error."""
    import yaml as _yaml

    templates_path = _CONTRIB_DIR / "graph_templates.yaml"
    if not templates_path.exists():
        return CheckResult(
            name="Graph templates",
            passed=False,
            critical=True,
            message="graph_templates.yaml not found",
        )

    try:
        with open(templates_path) as f:
            data = _yaml.safe_load(f)
        templates = data.get("templates", {})
    except Exception as exc:
        return CheckResult(
            name="Graph templates",
            passed=False,
            critical=True,
            message=f"YAML parse error: {exc}",
        )

    from pwm_core.graph.compiler import GraphCompiler
    from pwm_core.graph.graph_spec import OperatorGraphSpec

    compiler = GraphCompiler()
    errors = []
    compiled = 0
    for tid, tdata in templates.items():
        try:
            spec = OperatorGraphSpec(
                graph_id=tid,
                nodes=tdata["nodes"],
                edges=tdata["edges"],
                metadata=tdata.get("metadata", {}),
            )
            compiler.compile(spec)
            compiled += 1
        except Exception as exc:
            errors.append(f"{tid}: {exc}")

    total = len(templates)
    ok = compiled == total
    return CheckResult(
        name="Graph templates",
        passed=ok,
        critical=True,
        message=f"{compiled}/{total} compile OK",
        details=errors[:5],  # limit detail output
    )


def check_casepacks() -> CheckResult:
    """Check at least 1 CasePack is discoverable."""
    if not _CASEPACKS_DIR.is_dir():
        return CheckResult(
            name="CasePacks",
            passed=False,
            critical=True,
            message="casepacks/ directory not found",
        )
    packs = list(_CASEPACKS_DIR.glob("*.json"))
    ok = len(packs) > 0
    return CheckResult(
        name="CasePacks",
        passed=ok,
        critical=True,
        message=f"{len(packs)} found",
        details=[] if ok else ["No .json files in casepacks/"],
    )


def check_disk_write() -> CheckResult:
    """Check that a tmp directory is writable."""
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pwm_doctor") as f:
            f.write(b"pwm doctor check")
        return CheckResult(
            name="Disk/write",
            passed=True,
            critical=True,
            message="tmp writable",
        )
    except Exception as exc:
        return CheckResult(
            name="Disk/write",
            passed=False,
            critical=True,
            message=str(exc),
        )


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_python_version,
    check_core_deps,
    check_optional_deps,
    check_yaml_registries,
    check_graph_templates,
    check_casepacks,
    check_disk_write,
]


def run_doctor() -> List[CheckResult]:
    """Run all doctor checks and return results."""
    results = []
    for check_fn in ALL_CHECKS:
        try:
            results.append(check_fn())
        except Exception as exc:
            results.append(CheckResult(
                name=check_fn.__name__.replace("check_", "").replace("_", " ").title(),
                passed=False,
                critical=True,
                message=f"Unexpected error: {exc}",
            ))
    return results


def print_report(results: List[CheckResult]) -> int:
    """Print a formatted report and return exit code (0=ok, 1=fail)."""
    print()
    print(_bold("PWM Doctor"))
    print(_bold("=" * 50))
    print()

    any_critical_fail = False
    for r in results:
        tag = r.status_str
        print(f"  [{tag}] {r.name}: {r.message}")
        for d in r.details:
            print(f"         {d}")
        if not r.passed and r.critical:
            any_critical_fail = True

    print()
    if any_critical_fail:
        print(_red("Some critical checks failed. Fix the issues above."))
        return 1
    else:
        print(_green("All critical checks passed. Ready to go!"))
        return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cmd_doctor(args):
    """Entry point for ``pwm doctor``."""
    results = run_doctor()
    exit_code = print_report(results)
    raise SystemExit(exit_code)


def add_doctor_subparser(subparsers):
    """Add the doctor subcommand to the CLI parser."""
    p_doc = subparsers.add_parser(
        "doctor",
        help="Check environment, dependencies, and registries",
    )
    p_doc.set_defaults(func=cmd_doctor)
    return p_doc
