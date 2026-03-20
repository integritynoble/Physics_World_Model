"""Comprehensive tests for community/validate.py — all functions."""
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from community.validate import (
    REQUIRED_ARTIFACTS,
    REQUIRED_METRICS,
    REQUIRED_PROVENANCE,
    REQUIRED_TOP_LEVEL,
    ValidationResult,
    _is_finite_float,
    _parse_iso8601,
    sha256_of_file,
    validate_artifact_files,
    validate_manifest,
    validate_runbundle,
    validate_zip,
)
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────

def _valid_manifest():
    return {
        "version": "0.3.0",
        "spec_id": "cassi-scene1",
        "timestamp": "2026-02-21T12:00:00Z",
        "provenance": {
            "git_hash": "abc1234567",
            "seeds": [42],
            "platform": "linux",
            "pwm_version": "0.4",
        },
        "metrics": {"psnr_db": 32.5, "ssim": 0.95, "runtime_s": 1.2},
        "artifacts": {
            "x_gt": "x_gt.npy",
            "y": "y.npy",
            "x_hat": "x_hat.npy",
        },
        "hashes": {
            "x_gt": "sha256:aaa",
            "y": "sha256:bbb",
            "x_hat": "sha256:ccc",
        },
    }


# ── ValidationResult ────────────────────────────────────────────────

class TestValidationResult:
    def test_empty_passes(self):
        r = ValidationResult()
        assert r.passed
        assert "PASS" in r.summary()

    def test_error_fails(self):
        r = ValidationResult()
        r.error("oops")
        assert not r.passed
        assert "FAIL" in r.summary()
        assert "oops" in r.summary()

    def test_warning_still_passes(self):
        r = ValidationResult()
        r.warn("hmm")
        assert r.passed
        assert "WARNING" in r.summary()


# ── sha256_of_file ──────────────────────────────────────────────────

class TestSha256:
    def test_correct_hash(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        assert sha256_of_file(f) == hashlib.sha256(b"hello world").hexdigest()


# ── _is_finite_float ────────────────────────────────────────────────

class TestIsFiniteFloat:
    @pytest.mark.parametrize("val", [3.14, 0, -1.5, 100])
    def test_finite(self, val):
        assert _is_finite_float(val)

    @pytest.mark.parametrize("val", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite(self, val):
        assert not _is_finite_float(val)

    @pytest.mark.parametrize("val", ["abc", None, [], {}])
    def test_not_numeric(self, val):
        assert not _is_finite_float(val)


# ── _parse_iso8601 ──────────────────────────────────────────────────

class TestParseISO8601:
    @pytest.mark.parametrize(
        "ts",
        [
            "2026-02-21T12:00:00Z",
            "2026-02-21T12:00:00",
            "2026-02-21T12:00:00.123Z",
        ],
    )
    def test_valid(self, ts):
        assert _parse_iso8601(ts) is not None

    @pytest.mark.parametrize("ts", ["not a date", "", "2026-13-40"])
    def test_invalid(self, ts):
        assert _parse_iso8601(ts) is None


# ── validate_manifest ───────────────────────────────────────────────

class TestValidateManifest:
    def test_valid(self):
        r = ValidationResult()
        validate_manifest(_valid_manifest(), r)
        assert r.passed, r.errors

    def test_empty(self):
        r = ValidationResult()
        validate_manifest({}, r)
        assert not r.passed
        assert len(r.errors) == len(REQUIRED_TOP_LEVEL)

    def test_bad_version(self):
        r = ValidationResult()
        m = _valid_manifest()
        m["version"] = "999.0"
        validate_manifest(m, r)
        assert any("version" in e.lower() for e in r.errors)

    def test_bad_timestamp(self):
        r = ValidationResult()
        m = _valid_manifest()
        m["timestamp"] = "not-a-date"
        validate_manifest(m, r)
        assert any("timestamp" in e.lower() for e in r.errors)

    def test_empty_spec_id(self):
        r = ValidationResult()
        m = _valid_manifest()
        m["spec_id"] = ""
        validate_manifest(m, r)
        assert any("spec_id" in e for e in r.errors)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_metric(self, bad):
        r = ValidationResult()
        m = _valid_manifest()
        m["metrics"]["psnr_db"] = bad
        validate_manifest(m, r)
        assert any("psnr_db" in e for e in r.errors)

    def test_missing_provenance_fields(self):
        r = ValidationResult()
        m = _valid_manifest()
        m["provenance"] = {"git_hash": "abc1234"}
        validate_manifest(m, r)
        missing = REQUIRED_PROVENANCE - {"git_hash"}
        assert len([e for e in r.errors if "provenance" in e.lower()]) >= len(missing)

    def test_empty_seeds(self):
        r = ValidationResult()
        m = _valid_manifest()
        m["provenance"]["seeds"] = []
        validate_manifest(m, r)
        assert any("seeds" in e for e in r.errors)

    def test_short_git_hash(self):
        r = ValidationResult()
        m = _valid_manifest()
        m["provenance"]["git_hash"] = "abc"
        validate_manifest(m, r)
        assert any("git_hash" in e for e in r.errors)

    def test_missing_artifact(self):
        r = ValidationResult()
        m = _valid_manifest()
        m["artifacts"] = {"x_gt": "x_gt.npy"}
        validate_manifest(m, r)
        assert any("x_hat" in e for e in r.errors)

    def test_bad_hash_prefix(self):
        r = ValidationResult()
        m = _valid_manifest()
        m["hashes"]["x_gt"] = "md5:abc"
        validate_manifest(m, r)
        assert any("sha256" in e for e in r.errors)

    def test_missing_hash_for_artifact(self):
        r = ValidationResult()
        m = _valid_manifest()
        del m["hashes"]["x_gt"]
        validate_manifest(m, r)
        assert any("x_gt" in e for e in r.errors)

    def test_provenance_not_dict(self):
        r = ValidationResult()
        m = _valid_manifest()
        m["provenance"] = "string"
        validate_manifest(m, r)
        assert any("provenance" in e for e in r.errors)

    def test_metrics_not_dict(self):
        r = ValidationResult()
        m = _valid_manifest()
        m["metrics"] = "string"
        validate_manifest(m, r)
        assert any("metrics" in e for e in r.errors)

    def test_artifacts_not_dict(self):
        r = ValidationResult()
        m = _valid_manifest()
        m["artifacts"] = "string"
        validate_manifest(m, r)
        assert any("artifacts" in e for e in r.errors)


# ── validate_artifact_files ─────────────────────────────────────────

class TestValidateArtifactFiles:
    def test_matching_hash(self, tmp_path):
        np.save(tmp_path / "x_gt.npy", np.zeros(10))
        actual = sha256_of_file(tmp_path / "x_gt.npy")
        r = ValidationResult()
        validate_artifact_files(
            {"artifacts": {"x_gt": "x_gt.npy"}, "hashes": {"x_gt": f"sha256:{actual}"}},
            tmp_path,
            r,
        )
        assert r.passed

    def test_mismatching_hash(self, tmp_path):
        np.save(tmp_path / "x_gt.npy", np.zeros(10))
        r = ValidationResult()
        validate_artifact_files(
            {"artifacts": {"x_gt": "x_gt.npy"}, "hashes": {"x_gt": "sha256:000000"}},
            tmp_path,
            r,
        )
        assert any("mismatch" in e.lower() for e in r.errors)

    def test_missing_file(self, tmp_path):
        r = ValidationResult()
        validate_artifact_files(
            {"artifacts": {"x": "nope.npy"}, "hashes": {"x": "sha256:000"}},
            tmp_path,
            r,
        )
        assert any("not found" in e.lower() for e in r.errors)


# ── validate_runbundle ──────────────────────────────────────────────

class TestValidateRunbundle:
    def test_valid_bundle(self, tmp_path):
        np.save(tmp_path / "x_gt.npy", np.zeros(5))
        np.save(tmp_path / "y.npy", np.zeros(5))
        np.save(tmp_path / "x_hat.npy", np.zeros(5))
        m = _valid_manifest()
        m["hashes"] = {
            "x_gt": f"sha256:{sha256_of_file(tmp_path / 'x_gt.npy')}",
            "y": f"sha256:{sha256_of_file(tmp_path / 'y.npy')}",
            "x_hat": f"sha256:{sha256_of_file(tmp_path / 'x_hat.npy')}",
        }
        with open(tmp_path / "runbundle_manifest.json", "w") as f:
            json.dump(m, f)
        result = validate_runbundle(tmp_path)
        assert result.passed, result.errors

    def test_no_manifest(self, tmp_path):
        result = validate_runbundle(tmp_path)
        assert not result.passed

    def test_nested_manifest(self, tmp_path):
        sub = tmp_path / "inner"
        sub.mkdir()
        np.save(sub / "x_gt.npy", np.zeros(5))
        np.save(sub / "y.npy", np.zeros(5))
        np.save(sub / "x_hat.npy", np.zeros(5))
        m = _valid_manifest()
        m["hashes"] = {
            "x_gt": f"sha256:{sha256_of_file(sub / 'x_gt.npy')}",
            "y": f"sha256:{sha256_of_file(sub / 'y.npy')}",
            "x_hat": f"sha256:{sha256_of_file(sub / 'x_hat.npy')}",
        }
        with open(sub / "runbundle_manifest.json", "w") as f:
            json.dump(m, f)
        result = validate_runbundle(tmp_path)
        assert result.passed, result.errors

    def test_invalid_json(self, tmp_path):
        (tmp_path / "runbundle_manifest.json").write_text("{bad json")
        result = validate_runbundle(tmp_path)
        assert not result.passed


# ── validate_zip ────────────────────────────────────────────────────

class TestValidateZip:
    def test_not_a_file(self):
        r = validate_zip(Path("/nonexistent.zip"))
        assert not r.passed

    def test_not_a_zip(self, tmp_path):
        f = tmp_path / "fake.zip"
        f.write_text("not a zip")
        r = validate_zip(f)
        assert not r.passed
