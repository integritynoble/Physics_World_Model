"""test_demo_command.py

Tests for the ``pwm demo`` CLI subcommand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import pytest


def test_demo_parser_accepts_valid_args():
    """The demo subparser should accept modality + optional flags."""
    from pwm_core.cli.main import build_parser

    parser = build_parser()

    # Basic modality only
    args = parser.parse_args(["demo", "cassi"])
    assert args.modality == "cassi"
    assert args.run is False
    assert args.export_sharepack is False
    assert args.open_viewer is False
    assert args.preset is None

    # Full flags
    args = parser.parse_args([
        "demo", "mri", "--preset", "brain_t1", "--run",
        "--export-sharepack", "--open-viewer",
    ])
    assert args.modality == "mri"
    assert args.preset == "brain_t1"
    assert args.run is True
    assert args.export_sharepack is True
    assert args.open_viewer is True


def test_demo_parser_requires_modality():
    """The demo subparser should fail without a modality argument."""
    from pwm_core.cli.main import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["demo"])


def test_cmd_demo_info_mode(capsys):
    """cmd_demo without --run should print CasePack info as JSON."""
    from pwm_core.cli.demo import cmd_demo

    # Create a mock args namespace
    args = argparse.Namespace(
        modality="cassi",
        preset=None,
        run=False,
        export_sharepack=False,
        open_viewer=False,
    )

    # Should not raise
    cmd_demo(args)

    captured = capsys.readouterr()
    # Should output JSON with casepack info
    data = json.loads(captured.out)
    assert data["modality"] == "cassi"
    assert "casepack_id" in data


def test_cmd_demo_run_mode(tmp_path):
    """cmd_demo with --run should produce a RunBundle."""
    from pwm_core.cli.demo import cmd_demo

    import os
    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        args = argparse.Namespace(
            modality="cassi",
            preset=None,
            run=True,
            export_sharepack=False,
            open_viewer=False,
        )
        cmd_demo(args)

        # Check that a RunBundle was created
        rb_dir = tmp_path / "runs" / "demo_cassi" / "runbundle"
        assert rb_dir.exists()
        assert (rb_dir / "runbundle_manifest.json").exists()
    finally:
        os.chdir(original_cwd)


def test_cmd_demo_with_sharepack(tmp_path):
    """cmd_demo with --run --export-sharepack should produce sharepack."""
    from pwm_core.cli.demo import cmd_demo

    import os
    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        args = argparse.Namespace(
            modality="widefield",
            preset=None,
            run=True,
            export_sharepack=True,
            open_viewer=False,
        )
        cmd_demo(args)

        sp_dir = tmp_path / "runs" / "demo_widefield" / "sharepack"
        assert sp_dir.exists()
        assert (sp_dir / "summary.md").exists()
        assert (sp_dir / "metrics.json").exists()
        assert (sp_dir / "reproduce.sh").exists()
    finally:
        os.chdir(original_cwd)


def test_load_casepack_valid():
    """_load_casepack should find casepacks for known modalities."""
    from pwm_core.cli.demo import _load_casepack

    cp = _load_casepack("cassi")
    assert cp["modality"] == "cassi"
    assert "base_spec" in cp


def test_load_casepack_invalid():
    """_load_casepack should raise SystemExit for unknown modalities."""
    from pwm_core.cli.demo import _load_casepack

    with pytest.raises(SystemExit):
        _load_casepack("nonexistent_modality_xyz")
