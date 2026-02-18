"""test_dataset_infra.py

Tests for Track D: Dataset Infrastructure.

Tests
-----
* Manifest validation (valid + invalid)
* SHA256 computation and verification
* Dataset retrieval to cache
* All manifests validation (if manifests exist)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pwm_core.io.manifest import (
    validate_manifest,
    validate_manifest_file,
    validate_all_manifests,
    REQUIRED_FIELDS,
)
from pwm_core.io.retrieval import (
    compute_sha256,
    verify_file,
    retrieve_dataset,
)


# ---------------------------------------------------------------------------
# D.1: Manifest validation
# ---------------------------------------------------------------------------

class TestManifestValidation:

    def test_validate_manifest_valid(self):
        """A complete manifest should pass validation."""
        manifest = {
            "dataset_id": "test_v1",
            "modality": "cassi",
            "license": "MIT",
            "samples": [
                {"id": "s001", "path": "/data/s001.npy", "split": "train"},
                {"id": "s002", "path": "/data/s002.npy", "split": "test"},
            ],
        }
        errors = validate_manifest(manifest)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_validate_manifest_missing_fields(self):
        """Missing required fields should produce errors."""
        manifest = {"dataset_id": "incomplete"}
        errors = validate_manifest(manifest)
        assert len(errors) >= 2  # missing modality, samples, license
        assert any("modality" in e for e in errors)
        assert any("samples" in e for e in errors)
        assert any("license" in e for e in errors)

    def test_validate_manifest_bad_samples(self):
        """Samples missing required fields should produce errors."""
        manifest = {
            "dataset_id": "test",
            "modality": "spc",
            "license": "MIT",
            "samples": [
                {"id": "s001"},  # missing path and split
            ],
        }
        errors = validate_manifest(manifest)
        assert any("path" in e for e in errors)
        assert any("split" in e for e in errors)

    def test_validate_manifest_bad_split(self):
        """Invalid split value should produce error."""
        manifest = {
            "dataset_id": "test",
            "modality": "spc",
            "license": "MIT",
            "samples": [
                {"id": "s001", "path": "/data/s.npy", "split": "invalid"},
            ],
        }
        errors = validate_manifest(manifest)
        assert any("split" in e for e in errors)

    def test_validate_manifest_file_valid(self, tmp_path):
        """Validate a manifest from a JSON file."""
        manifest = {
            "dataset_id": "file_test",
            "modality": "ct",
            "license": "CC-BY-4.0",
            "samples": [
                {"id": "s1", "path": "s1.npy", "split": "train"},
            ],
        }
        fpath = str(tmp_path / "manifest.json")
        with open(fpath, "w") as f:
            json.dump(manifest, f)

        errors = validate_manifest_file(fpath)
        assert errors == []

    def test_validate_manifest_file_not_found(self):
        """Non-existent file should return error."""
        errors = validate_manifest_file("/nonexistent/manifest.json")
        assert len(errors) == 1
        assert "not found" in errors[0].lower() or "File not found" in errors[0]

    def test_validate_all_manifests(self, tmp_path):
        """Validate all manifests in a directory."""
        # Create two manifest files
        for name in ["a.json", "b.json"]:
            manifest = {
                "dataset_id": name.replace(".json", ""),
                "modality": "spc",
                "license": "MIT",
                "samples": [
                    {"id": "s1", "path": "s1.npy", "split": "train"},
                ],
            }
            with open(str(tmp_path / name), "w") as f:
                json.dump(manifest, f)

        results = validate_all_manifests(str(tmp_path))
        assert len(results) == 2
        for fname, errs in results.items():
            assert errs == [], f"{fname} has errors: {errs}"


# ---------------------------------------------------------------------------
# D.2: SHA256 and retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:

    def test_compute_sha256(self, tmp_path):
        """Known content should produce known digest."""
        fpath = str(tmp_path / "test.bin")
        content = b"hello world"
        with open(fpath, "wb") as f:
            f.write(content)

        digest = compute_sha256(fpath)
        import hashlib
        expected = hashlib.sha256(content).hexdigest()
        assert digest == expected

    def test_verify_file_correct(self, tmp_path):
        """Matching hash should return True."""
        fpath = str(tmp_path / "good.bin")
        content = b"test data"
        with open(fpath, "wb") as f:
            f.write(content)

        import hashlib
        expected = hashlib.sha256(content).hexdigest()
        assert verify_file(fpath, expected) is True

    def test_verify_file_corrupt(self, tmp_path):
        """Wrong hash should return False."""
        fpath = str(tmp_path / "bad.bin")
        with open(fpath, "wb") as f:
            f.write(b"actual content")

        assert verify_file(fpath, "0" * 64) is False

    def test_verify_file_missing(self):
        """Missing file should return False."""
        assert verify_file("/nonexistent/file.bin", "abc") is False

    def test_retrieve_dataset(self, tmp_path):
        """Retrieve dataset copies files to cache and verifies."""
        # Create source files
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        sample_data = np.random.default_rng(42).random((8, 8)).astype(np.float32)
        sample_path = src_dir / "sample.npy"
        np.save(str(sample_path), sample_data)

        sha = compute_sha256(str(sample_path))

        manifest = {
            "dataset_id": "retrieval_test",
            "modality": "spc",
            "license": "MIT",
            "samples": [
                {
                    "id": "s1",
                    "path": str(sample_path),
                    "split": "train",
                    "sha256": sha,
                },
            ],
        }
        manifest_path = str(src_dir / "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        cache_dir = str(tmp_path / "cache")
        result = retrieve_dataset(manifest_path, cache_dir, verify=True)

        assert os.path.isdir(result)
        assert os.path.exists(os.path.join(result, "sample.npy"))

    def test_retrieve_dataset_bad_hash(self, tmp_path):
        """Retrieval should raise ValueError on hash mismatch."""
        src_dir = tmp_path / "source2"
        src_dir.mkdir()
        sample_path = src_dir / "sample.npy"
        np.save(str(sample_path), np.zeros((4, 4)))

        manifest = {
            "dataset_id": "bad_hash_test",
            "modality": "spc",
            "license": "MIT",
            "samples": [
                {
                    "id": "s1",
                    "path": str(sample_path),
                    "split": "train",
                    "sha256": "0" * 64,  # wrong hash
                },
            ],
        }
        manifest_path = str(src_dir / "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        cache_dir = str(tmp_path / "cache2")
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            retrieve_dataset(manifest_path, cache_dir, verify=True)
