"""End-to-end tests for PWM challenge scoring pipeline.

Tests the full flow: create mock submission -> validate -> score -> leaderboard.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

# Ensure community/ is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import the modules under test
from community.validate import (
    MANIFEST_FILENAME,
    RUNBUNDLE_VERSION,
    ValidationResult,
    validate_manifest,
    validate_runbundle,
    validate_zip,
)
from community.leaderboard import (
    ScoredSubmission,
    collect_submissions,
    generate_leaderboard_md,
    load_expected,
    score_submission,
)


# ---- Helpers ----


def sha256_of_bytes(data: bytes) -> str:
    """Compute SHA256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def make_valid_manifest(
    artifact_hashes: dict[str, str] | None = None,
) -> dict:
    """Create a valid RunBundle manifest dict."""
    return {
        "version": "0.3.0",
        "spec_id": "test_run_001",
        "timestamp": "2026-02-09T12:00:00Z",
        "provenance": {
            "git_hash": "abc1234def",
            "seeds": [42],
            "platform": "linux-x86_64",
            "pwm_version": "0.3.0",
        },
        "metrics": {
            "psnr_db": 30.5,
            "ssim": 0.92,
            "runtime_s": 12.3,
        },
        "artifacts": {
            "x_gt": "data/x_gt.npy",
            "y": "data/y.npy",
            "x_hat": "results/x_hat.npy",
        },
        "hashes": artifact_hashes
        or {
            "x_gt": "sha256:placeholder",
            "y": "sha256:placeholder",
            "x_hat": "sha256:placeholder",
        },
    }


def create_mock_runbundle(
    bundle_dir: Path,
    manifest_overrides: dict | None = None,
    corrupt_hash: bool = False,
) -> Path:
    """Create a mock RunBundle directory with artifacts and correct hashes.

    Args:
        bundle_dir: Directory to create the RunBundle in.
        manifest_overrides: Optional dict to merge into the manifest.
        corrupt_hash: If True, put wrong hash for x_hat.

    Returns:
        Path to the bundle directory.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    data_dir = bundle_dir / "data"
    results_dir = bundle_dir / "results"
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Create artifact files
    rng = np.random.RandomState(42)
    x_gt = rng.rand(32, 32).astype(np.float64)
    y = rng.rand(32, 32).astype(np.float64)
    x_hat = x_gt + rng.randn(32, 32) * 0.05  # noisy reconstruction

    np.save(data_dir / "x_gt.npy", x_gt)
    np.save(data_dir / "y.npy", y)
    np.save(results_dir / "x_hat.npy", x_hat)

    # Compute hashes
    hashes = {}
    for key, path in [
        ("x_gt", data_dir / "x_gt.npy"),
        ("y", data_dir / "y.npy"),
        ("x_hat", results_dir / "x_hat.npy"),
    ]:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        hashes[key] = f"sha256:{h.hexdigest()}"

    if corrupt_hash:
        hashes["x_hat"] = "sha256:0000000000000000000000000000000000000000"

    manifest = make_valid_manifest(artifact_hashes=hashes)
    if manifest_overrides:
        manifest.update(manifest_overrides)

    with open(bundle_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f, indent=2)

    return bundle_dir


def create_mock_zip(tmp_path: Path, **kwargs) -> Path:
    """Create a mock RunBundle as a zip file."""
    bundle_dir = tmp_path / "bundle"
    create_mock_runbundle(bundle_dir, **kwargs)

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(str(zip_path), "w") as zf:
        for root, _dirs, files in os.walk(str(bundle_dir)):
            for fname in files:
                abs_path = Path(root) / fname
                arc_name = abs_path.relative_to(bundle_dir)
                zf.write(str(abs_path), str(arc_name))

    return zip_path


# ---- Validate Tests ----


class TestValidateManifest:
    """Tests for manifest-level validation."""

    def test_valid_manifest(self):
        manifest = make_valid_manifest()
        result = ValidationResult()
        validate_manifest(manifest, result)
        assert result.passed, result.summary()

    def test_missing_version(self):
        manifest = make_valid_manifest()
        del manifest["version"]
        result = ValidationResult()
        validate_manifest(manifest, result)
        assert not result.passed
        assert any("version" in e for e in result.errors)

    def test_wrong_version(self):
        manifest = make_valid_manifest()
        manifest["version"] = "0.1.0"
        result = ValidationResult()
        validate_manifest(manifest, result)
        assert not result.passed
        assert any("0.1.0" in e for e in result.errors)

    def test_missing_metrics(self):
        manifest = make_valid_manifest()
        del manifest["metrics"]["psnr_db"]
        result = ValidationResult()
        validate_manifest(manifest, result)
        assert not result.passed
        assert any("psnr_db" in e for e in result.errors)

    def test_nan_metric(self):
        manifest = make_valid_manifest()
        manifest["metrics"]["psnr_db"] = float("nan")
        result = ValidationResult()
        validate_manifest(manifest, result)
        assert not result.passed
        assert any("finite" in e.lower() for e in result.errors)

    def test_inf_metric(self):
        manifest = make_valid_manifest()
        manifest["metrics"]["ssim"] = float("inf")
        result = ValidationResult()
        validate_manifest(manifest, result)
        assert not result.passed

    def test_invalid_timestamp(self):
        manifest = make_valid_manifest()
        manifest["timestamp"] = "not-a-date"
        result = ValidationResult()
        validate_manifest(manifest, result)
        assert not result.passed
        assert any("ISO 8601" in e for e in result.errors)

    def test_empty_seeds(self):
        manifest = make_valid_manifest()
        manifest["provenance"]["seeds"] = []
        result = ValidationResult()
        validate_manifest(manifest, result)
        assert not result.passed
        assert any("seeds" in e for e in result.errors)

    def test_short_git_hash(self):
        manifest = make_valid_manifest()
        manifest["provenance"]["git_hash"] = "abc"
        result = ValidationResult()
        validate_manifest(manifest, result)
        assert not result.passed
        assert any("git_hash" in e for e in result.errors)

    def test_missing_artifact_key(self):
        manifest = make_valid_manifest()
        del manifest["artifacts"]["x_hat"]
        result = ValidationResult()
        validate_manifest(manifest, result)
        assert not result.passed
        assert any("x_hat" in e for e in result.errors)

    def test_hash_without_sha256_prefix(self):
        manifest = make_valid_manifest()
        manifest["hashes"]["x_gt"] = "md5:abc123"
        result = ValidationResult()
        validate_manifest(manifest, result)
        assert not result.passed
        assert any("sha256:" in e for e in result.errors)

    def test_missing_hash_for_artifact(self):
        manifest = make_valid_manifest()
        del manifest["hashes"]["y"]
        result = ValidationResult()
        validate_manifest(manifest, result)
        assert not result.passed
        assert any("'y'" in e for e in result.errors)


class TestValidateRunBundle:
    """Tests for full RunBundle directory validation."""

    def test_valid_bundle(self, tmp_path):
        bundle_dir = create_mock_runbundle(tmp_path / "bundle")
        result = validate_runbundle(bundle_dir)
        assert result.passed, result.summary()

    def test_corrupt_hash(self, tmp_path):
        bundle_dir = create_mock_runbundle(
            tmp_path / "bundle", corrupt_hash=True
        )
        result = validate_runbundle(bundle_dir)
        assert not result.passed
        assert any("Hash mismatch" in e for e in result.errors)

    def test_missing_artifact_file(self, tmp_path):
        bundle_dir = create_mock_runbundle(tmp_path / "bundle")
        # Delete an artifact
        (bundle_dir / "results" / "x_hat.npy").unlink()
        result = validate_runbundle(bundle_dir)
        assert not result.passed
        assert any("not found" in e for e in result.errors)

    def test_no_manifest(self, tmp_path):
        bundle_dir = tmp_path / "empty_bundle"
        bundle_dir.mkdir()
        result = validate_runbundle(bundle_dir)
        assert not result.passed
        assert any("not found" in e for e in result.errors)


class TestValidateZip:
    """Tests for zip file validation."""

    def test_valid_zip(self, tmp_path):
        zip_path = create_mock_zip(tmp_path)
        result = validate_zip(zip_path)
        assert result.passed, result.summary()

    def test_corrupt_hash_zip(self, tmp_path):
        zip_path = create_mock_zip(tmp_path, corrupt_hash=True)
        result = validate_zip(zip_path)
        assert not result.passed

    def test_nonexistent_file(self, tmp_path):
        result = validate_zip(tmp_path / "nonexistent.zip")
        assert not result.passed

    def test_not_a_zip(self, tmp_path):
        fake_zip = tmp_path / "fake.zip"
        fake_zip.write_text("this is not a zip file")
        result = validate_zip(fake_zip)
        assert not result.passed


# ---- Score Tests ----


class TestScoreSubmission:
    """Tests for scoring submissions against expected metrics."""

    @pytest.fixture
    def expected(self):
        return {
            "challenge_version": "1.0.0",
            "week_id": "2026-W10",
            "modality": "cassi",
            "primary_metric": "psnr_db",
            "secondary_metric": "ssim",
            "reference_metrics": {
                "psnr_db": 30.6,
                "ssim": 0.91,
                "runtime_s": 12.0,
            },
            "thresholds": {
                "psnr_db_min": 20.0,
                "ssim_min": 0.5,
                "runtime_s_max": 300.0,
            },
        }

    def test_valid_submission(self, expected):
        manifest = make_valid_manifest()
        sub = score_submission(manifest, expected, name="team_alpha")
        assert sub.valid
        assert sub.psnr_db == 30.5
        assert sub.ssim == 0.92
        assert sub.name == "team_alpha"

    def test_below_psnr_threshold(self, expected):
        manifest = make_valid_manifest()
        manifest["metrics"]["psnr_db"] = 15.0
        sub = score_submission(manifest, expected, name="low_psnr")
        assert not sub.valid
        assert "PSNR" in sub.invalid_reason

    def test_below_ssim_threshold(self, expected):
        manifest = make_valid_manifest()
        manifest["metrics"]["ssim"] = 0.3
        sub = score_submission(manifest, expected, name="low_ssim")
        assert not sub.valid
        assert "SSIM" in sub.invalid_reason

    def test_above_runtime_threshold(self, expected):
        manifest = make_valid_manifest()
        manifest["metrics"]["runtime_s"] = 500.0
        sub = score_submission(manifest, expected, name="slow")
        assert not sub.valid
        assert "Runtime" in sub.invalid_reason

    def test_nan_metric_invalid(self, expected):
        manifest = make_valid_manifest()
        manifest["metrics"]["psnr_db"] = float("nan")
        sub = score_submission(manifest, expected, name="nan_psnr")
        assert not sub.valid
        assert "finite" in sub.invalid_reason

    def test_theta_error_included(self, expected):
        manifest = make_valid_manifest()
        manifest["metrics"]["theta_error_rmse"] = 0.05
        sub = score_submission(manifest, expected, name="with_theta")
        assert sub.valid
        assert sub.theta_error == 0.05

    def test_ranking_order(self, expected):
        manifests = [
            {"psnr_db": 25.0, "ssim": 0.8, "runtime_s": 10.0},
            {"psnr_db": 32.0, "ssim": 0.95, "runtime_s": 20.0},
            {"psnr_db": 28.0, "ssim": 0.85, "runtime_s": 5.0},
        ]
        subs = []
        for i, m in enumerate(manifests):
            manifest = make_valid_manifest()
            manifest["metrics"] = m
            subs.append(score_submission(manifest, expected, name=f"team_{i}"))

        subs.sort(key=lambda s: s.rank_key, reverse=True)
        # team_1 (32 dB) should be first
        assert subs[0].psnr_db == 32.0
        assert subs[1].psnr_db == 28.0
        assert subs[2].psnr_db == 25.0


# ---- Leaderboard Tests ----


class TestLeaderboard:
    """Tests for leaderboard generation."""

    @pytest.fixture
    def expected(self):
        return {
            "challenge_version": "1.0.0",
            "week_id": "2026-W10",
            "modality": "cassi",
            "primary_metric": "psnr_db",
            "secondary_metric": "ssim",
            "reference_metrics": {
                "psnr_db": 30.6,
                "ssim": 0.91,
                "runtime_s": 12.0,
            },
            "thresholds": {
                "psnr_db_min": 20.0,
                "ssim_min": 0.5,
                "runtime_s_max": 300.0,
            },
        }

    def test_empty_leaderboard(self, expected):
        md = generate_leaderboard_md("2026-W10", expected, [])
        assert "2026-W10" in md
        assert "No submissions" in md

    def test_leaderboard_with_submissions(self, expected):
        subs = [
            ScoredSubmission(
                name="team_a", spec_id="run_a", psnr_db=31.0,
                ssim=0.93, runtime_s=10.0, valid=True,
            ),
            ScoredSubmission(
                name="team_b", spec_id="run_b", psnr_db=28.0,
                ssim=0.85, runtime_s=20.0, valid=True,
            ),
            ScoredSubmission(
                name="team_c", spec_id="run_c", psnr_db=15.0,
                ssim=0.3, runtime_s=5.0, valid=False,
                invalid_reason="PSNR below minimum",
            ),
        ]
        md = generate_leaderboard_md("2026-W10", expected, subs)
        assert "Rankings" in md
        assert "team_a" in md
        assert "team_b" in md
        assert "INVALID" in md
        assert "31.00" in md
        assert "2026-W10" in md

    def test_collect_submissions_empty(self, tmp_path):
        subs = collect_submissions(tmp_path / "nonexistent")
        assert subs == []

    def test_collect_submissions_from_dirs(self, tmp_path):
        subs_dir = tmp_path / "submissions"
        # Create two submission directories
        for name in ["team_a", "team_b"]:
            sub_dir = subs_dir / name
            create_mock_runbundle(sub_dir)

        results = collect_submissions(subs_dir)
        assert len(results) == 2
        names = {r[0] for r in results}
        assert "team_a" in names
        assert "team_b" in names


class TestLoadExpected:
    """Tests for loading expected.json from challenge directories."""

    def test_load_existing_week(self):
        """Load the expected.json for 2026-W10 (should exist in repo)."""
        expected = load_expected("2026-W10")
        assert expected["week_id"] == "2026-W10"
        assert expected["modality"] == "cassi"
        assert "reference_metrics" in expected

    def test_load_nonexistent_week(self):
        with pytest.raises(FileNotFoundError):
            load_expected("9999-W99")


# ---- End-to-End Tests ----


class TestEndToEnd:
    """Full pipeline: mock submission -> validate -> score -> leaderboard."""

    def test_full_pipeline(self, tmp_path):
        """Create a mock submission, validate it, score it, generate leaderboard."""
        # 1. Create mock submission zip
        zip_path = create_mock_zip(tmp_path)
        assert zip_path.is_file()

        # 2. Validate
        result = validate_zip(zip_path)
        assert result.passed, result.summary()

        # 3. Load expected
        expected = load_expected("2026-W10")

        # 4. Score
        manifest = make_valid_manifest()
        sub = score_submission(manifest, expected, name="test_team")
        assert sub.valid
        assert sub.psnr_db == 30.5

        # 5. Generate leaderboard
        md = generate_leaderboard_md("2026-W10", expected, [sub])
        assert "test_team" in md
        assert "30.50" in md

        # 6. Write leaderboard
        lb_path = tmp_path / "leaderboard.md"
        lb_path.write_text(md)
        assert lb_path.is_file()
        assert lb_path.stat().st_size > 0

    def test_invalid_submission_pipeline(self, tmp_path):
        """A submission with corrupt hash should fail validation but still score."""
        zip_path = create_mock_zip(tmp_path, corrupt_hash=True)
        result = validate_zip(zip_path)
        assert not result.passed

        # Even if validation fails, scoring can still run on manifest data
        manifest = make_valid_manifest()
        expected = load_expected("2026-W10")
        sub = score_submission(manifest, expected, name="corrupt_team")
        assert sub.valid  # Metrics are still valid; hash issue is separate
