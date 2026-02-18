"""test_doctor.py

Tests for the ``pwm doctor`` CLI subcommand.
"""
from __future__ import annotations

import argparse

import pytest


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


def test_doctor_parser_accepts_command():
    """The parser should accept ``pwm doctor`` with no extra args."""
    from pwm_core.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["doctor"])
    assert args.cmd == "doctor"
    assert hasattr(args, "func")


# ---------------------------------------------------------------------------
# cmd_doctor integration test
# ---------------------------------------------------------------------------


def test_cmd_doctor_runs(capsys):
    """cmd_doctor should run all checks and print a report."""
    from pwm_core.cli.doctor import cmd_doctor

    args = argparse.Namespace()

    # cmd_doctor raises SystemExit; code 0 means all critical pass
    with pytest.raises(SystemExit) as exc_info:
        cmd_doctor(args)

    # In our repo all critical checks should pass (exit 0)
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "PWM Doctor" in captured.out
    assert "PASS" in captured.out


# ---------------------------------------------------------------------------
# Individual check function tests
# ---------------------------------------------------------------------------


def test_check_python_version():
    from pwm_core.cli.doctor import check_python_version

    result = check_python_version()
    assert result.name == "Python version"
    assert result.passed is True
    assert result.critical is True


def test_check_core_deps():
    from pwm_core.cli.doctor import check_core_deps

    result = check_core_deps()
    assert result.name == "Core dependencies"
    # In dev environment all core deps should be installed
    assert result.passed is True


def test_check_optional_deps():
    from pwm_core.cli.doctor import check_optional_deps

    result = check_optional_deps()
    assert result.name == "Optional dependencies"
    # Optional deps may or may not be installed — just check structure
    assert isinstance(result.passed, bool)
    assert result.critical is False


def test_check_yaml_registries():
    from pwm_core.cli.doctor import check_yaml_registries

    result = check_yaml_registries()
    assert result.name == "YAML registries"
    assert result.passed is True
    assert "6/6" in result.message


def test_check_graph_templates():
    from pwm_core.cli.doctor import check_graph_templates

    result = check_graph_templates()
    assert result.name == "Graph templates"
    assert result.passed is True
    assert "compile OK" in result.message


def test_check_casepacks():
    from pwm_core.cli.doctor import check_casepacks

    result = check_casepacks()
    assert result.name == "CasePacks"
    assert result.passed is True
    assert "found" in result.message


def test_check_disk_write():
    from pwm_core.cli.doctor import check_disk_write

    result = check_disk_write()
    assert result.name == "Disk/write"
    assert result.passed is True


# ---------------------------------------------------------------------------
# run_doctor returns structured results
# ---------------------------------------------------------------------------


def test_run_doctor_returns_list():
    from pwm_core.cli.doctor import run_doctor, CheckResult

    results = run_doctor()
    assert isinstance(results, list)
    assert len(results) == 7
    for r in results:
        assert isinstance(r, CheckResult)
        assert isinstance(r.name, str)
        assert isinstance(r.passed, bool)
