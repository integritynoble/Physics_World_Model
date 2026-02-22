import tempfile, os, hashlib, json, pathlib
from ingestion.checksums import sha256_file
from ingestion.manifest import generate_manifest
from ingestion.validator import validate_config


def test_sha256_correct():
    content = b"physics world model test data"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        fname = f.name
    try:
        result = sha256_file(fname)
        assert result == hashlib.sha256(content).hexdigest()
    finally:
        os.unlink(fname)


def test_manifest_contains_all_files():
    with tempfile.TemporaryDirectory() as dest:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                with open(os.path.join(tmpdir, f"sample_{i:04d}.bin"), "wb") as f:
                    f.write(f"data{i}".encode())
            manifest = generate_manifest(tmpdir, dest)
    assert len(manifest["files"]) == 5
    for entry in manifest["files"]:
        assert "sha256" in entry
        assert "dest_path" in entry
        assert entry["size_bytes"] > 0


def test_manifest_sha256_matches():
    content = b"deterministic content for pwm"
    with tempfile.TemporaryDirectory() as dest:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "file.bin"), "wb") as f:
                f.write(content)
            manifest = generate_manifest(tmpdir, dest)
    expected = hashlib.sha256(content).hexdigest()
    assert manifest["files"][0]["sha256"] == expected


def test_validate_config_missing_field():
    errors = validate_config({"name": "test"})
    assert any("dataset_id" in e for e in errors)


def test_validate_config_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        errors = validate_config({
            "dataset_id": "test_v1",
            "name": "Test",
            "modality": "ct",
            "kind": "simulation",
            "version": "1.0.0",
            "local_data_dir": tmpdir,
        })
    assert errors == []


def test_validate_config_bad_kind():
    with tempfile.TemporaryDirectory() as tmpdir:
        errors = validate_config({
            "dataset_id": "x", "name": "x", "modality": "ct",
            "kind": "bogus", "version": "1.0.0", "local_data_dir": tmpdir,
        })
    assert any("kind" in e for e in errors)
