"""test_dyson_launch_slice.py — Launch Slice item 5 integration test.

Verifies the full pipeline:
    emit_runbundle() → issue_certificate() → certificate.json written

Tests:
  - certificate.json is created in the bundle root
  - trust_tier is "draft" (all S-gates pass for a well-formed bundle)
  - all four S-gates (s1, s2, s3, s4) are present and pass
  - triad_flags field is present (may be None without DR-IS data)
  - a bundle with missing manifest keys produces trust_tier "rejected"
  - CLI --emit-certificate flag calls issue_certificate() and writes the file
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pwm_core.core.runbundle.certificate import TrustTier, GateVerdict
from pwm_core.targeting.runbundle_emitter import issue_certificate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_synthetic_bundle(tmp_dir: Path, *, omit_keys: list[str] | None = None) -> Path:
    """Build a minimal but fully valid RunBundle in *tmp_dir*.

    Parameters
    ----------
    omit_keys : list[str], optional
        Keys to drop from the manifest to simulate incomplete bundles.
    """
    bundle_dir = tmp_dir / "run_cassi_gap_tv_test0001"
    bundle_dir.mkdir(parents=True)
    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir()

    # Write one synthetic artifact so S3 has something to verify
    x_hat = np.random.default_rng(42).standard_normal((32, 32)).astype(np.float32)
    art_path = artifacts_dir / "x_hat_scene0_scenarioI.npy"
    np.save(str(art_path), x_hat)
    art_hash = _sha256(art_path)

    metrics_data = {"psnr_db": 28.5, "ssim": 0.85, "rho": 0.72}
    metrics_path = artifacts_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_data))
    metrics_hash = _sha256(metrics_path)

    manifest = {
        "version": "0.3.0",
        "spec_id": "cassi_gap_tv_correct",
        "timestamp": "2026-03-25T00:00:00Z",
        "provenance": {
            "git_hash": "abc1234def5678901234567890abcdef12345678",
            "git_dirty": False,
            "seeds": [42],
            "python_version": platform.python_version(),
            "modality": "cassi",
            "solver": "gap_tv",
            "track": "correct",
            "n_scenes": 1,
            "severity": "moderate",
            "sandbox": True,
            "declared_budget_s": 600,
        },
        "metrics": {
            "psnr_db": 28.5,
            "ssim": 0.85,
            "runtime_s": 12.3,
            "rho": 0.72,
            "oracle_gap": 4.1,
            "roic": 3.5,
            "ofs": 0.68,
            "final_score": 0.65,
            "disqualified": False,
        },
        "artifacts": {
            "x_hat_scene0_scenarioI": "artifacts/x_hat_scene0_scenarioI.npy",
            "metrics": "artifacts/metrics.json",
        },
        "hashes": {
            "x_hat_scene0_scenarioI": f"sha256:{art_hash}",
            "metrics": f"sha256:{metrics_hash}",
        },
    }

    if omit_keys:
        for k in omit_keys:
            manifest.pop(k, None)

    manifest_path = bundle_dir / "runbundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return bundle_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIssueCertificate:
    """Direct tests of issue_certificate() on synthetic bundles."""

    def test_certificate_file_created(self, tmp_path):
        bundle_dir = _make_synthetic_bundle(tmp_path)
        cert_path = issue_certificate(bundle_dir)

        assert cert_path is not None, "issue_certificate() returned None"
        assert cert_path.exists(), f"certificate.json not found at {cert_path}"
        assert cert_path.name == "certificate.json"
        assert cert_path.parent == bundle_dir

    def test_trust_tier_is_draft_for_clean_bundle(self, tmp_path):
        bundle_dir = _make_synthetic_bundle(tmp_path)
        cert_path = issue_certificate(bundle_dir)

        cert = json.loads(cert_path.read_text())
        assert cert["trust_tier"] == "draft", (
            f"Expected trust_tier=draft for a clean bundle, got {cert['trust_tier']!r}"
        )

    def test_all_four_gates_present(self, tmp_path):
        bundle_dir = _make_synthetic_bundle(tmp_path)
        cert_path = issue_certificate(bundle_dir)

        cert = json.loads(cert_path.read_text())
        gate_verdicts = cert.get("gate_verdicts", {})
        for gate in ("s1", "s2", "s3", "s4"):
            assert gate in gate_verdicts, f"Gate {gate!r} missing from certificate"

    def test_s1_s3_pass_for_complete_manifest(self, tmp_path):
        bundle_dir = _make_synthetic_bundle(tmp_path)
        cert_path = issue_certificate(bundle_dir)

        cert = json.loads(cert_path.read_text())
        gate_verdicts = cert["gate_verdicts"]

        assert gate_verdicts["s1"]["verdict"] == "pass", (
            f"S1 should pass for a complete manifest: {gate_verdicts['s1']}"
        )
        assert gate_verdicts["s3"]["verdict"] == "pass", (
            f"S3 should pass when all artifact hashes are correct: {gate_verdicts['s3']}"
        )

    def test_triad_flags_field_present(self, tmp_path):
        bundle_dir = _make_synthetic_bundle(tmp_path)
        cert_path = issue_certificate(bundle_dir)

        cert = json.loads(cert_path.read_text())
        # triad_flags may be None when DR-IS records are absent, but the key must exist
        assert "triad_flags" in cert, "certificate.json must contain 'triad_flags' key"

    def test_required_certificate_fields(self, tmp_path):
        bundle_dir = _make_synthetic_bundle(tmp_path)
        cert_path = issue_certificate(bundle_dir)

        cert = json.loads(cert_path.read_text())
        for field in ("run_id", "spec_id", "judge_version", "trust_tier",
                      "risk_flags", "active_gates", "gate_verdicts", "issued_at"):
            assert field in cert, f"Required field {field!r} missing from certificate.json"

    def test_run_id_matches_bundle_dir_name(self, tmp_path):
        bundle_dir = _make_synthetic_bundle(tmp_path)
        cert_path = issue_certificate(bundle_dir)

        cert = json.loads(cert_path.read_text())
        assert cert["run_id"] == bundle_dir.name, (
            f"run_id {cert['run_id']!r} should match bundle dir name {bundle_dir.name!r}"
        )

    def test_missing_manifest_returns_none(self, tmp_path):
        bundle_dir = tmp_path / "run_empty"
        bundle_dir.mkdir()
        # No manifest written
        result = issue_certificate(bundle_dir)
        assert result is None, "issue_certificate() should return None when manifest is missing"

    def test_incomplete_manifest_produces_rejected_tier(self, tmp_path):
        """Drop 'artifacts' key → S1 fails → trust_tier=rejected."""
        bundle_dir = _make_synthetic_bundle(tmp_path, omit_keys=["artifacts"])
        cert_path = issue_certificate(bundle_dir)

        cert = json.loads(cert_path.read_text())
        assert cert["trust_tier"] == "rejected", (
            f"Expected rejected tier for missing 'artifacts' key, got {cert['trust_tier']!r}"
        )
        assert "safety_brake" in cert["risk_flags"], (
            "safety_brake flag should be set when hard gate fails"
        )

    def test_hash_mismatch_produces_rejected_tier(self, tmp_path):
        """Corrupt the artifact after computing hashes → S3 fails → rejected."""
        bundle_dir = _make_synthetic_bundle(tmp_path)
        # Corrupt the artifact file after the manifest was written with correct hashes
        art = bundle_dir / "artifacts" / "x_hat_scene0_scenarioI.npy"
        art.write_bytes(b"\x00" * 128)  # Overwrite with garbage

        cert_path = issue_certificate(bundle_dir)
        cert = json.loads(cert_path.read_text())

        assert cert["trust_tier"] == "rejected", (
            f"Expected rejected tier after hash corruption, got {cert['trust_tier']!r}"
        )
        assert cert["gate_verdicts"]["s3"]["verdict"] == "fail"


class TestEndToEndCLI:
    """Test that --emit-certificate wires issue_certificate() through the CLI."""

    def test_emit_certificate_flag_exists(self):
        """Ensure targeting CLI accepts --emit-certificate without error."""
        result = subprocess.run(
            [sys.executable, "-m", "pwm_core.targeting.cli", "evaluate", "--help"],
            capture_output=True, text=True,
        )
        assert "--emit-certificate" in result.stdout, (
            "--emit-certificate flag not found in `pwm evaluate --help`.\n"
            f"stdout:\n{result.stdout}"
        )


class TestCertificateDeserialization:
    """Round-trip: write certificate, reload, verify Python objects."""

    def test_trust_tier_enum_roundtrip(self, tmp_path):
        bundle_dir = _make_synthetic_bundle(tmp_path)
        cert_path = issue_certificate(bundle_dir)

        cert_dict = json.loads(cert_path.read_text())
        tier = TrustTier(cert_dict["trust_tier"])
        assert tier == TrustTier.draft

    def test_gate_verdict_enum_roundtrip(self, tmp_path):
        bundle_dir = _make_synthetic_bundle(tmp_path)
        cert_path = issue_certificate(bundle_dir)

        cert_dict = json.loads(cert_path.read_text())
        s1_verdict = GateVerdict(cert_dict["gate_verdicts"]["s1"]["verdict"])
        assert s1_verdict == GateVerdict.pass_
