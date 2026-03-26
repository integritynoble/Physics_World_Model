"""test_gates.py -- R1-R4 gate function unit tests."""
import pytest
import hashlib
import numpy as np
from pathlib import Path
from pwm_core.targeting.gates import (
    check_r1_spec_completeness, check_r2_reproducibility,
    check_r3_metric_integrity, check_r4_budget_compliance,
    run_r1_r4, check_s1_spec_completeness  # backward alias
)
from pwm_core.core.runbundle.certificate import GateVerdict

class TestR1SpecCompleteness:
    def test_pass(self):
        manifest = {"version": "0.3.0", "spec_id": "test", "timestamp": "2026-01-01",
                     "provenance": {"modality": "ct", "solver": "fbp"}, "metrics": {}, "artifacts": {}}
        r = check_r1_spec_completeness(manifest)
        assert r.verdict == GateVerdict.pass_

    def test_fail_missing_keys(self):
        r = check_r1_spec_completeness({"version": "0.3.0"})
        assert r.verdict == GateVerdict.fail

class TestR2Reproducibility:
    def test_pass(self):
        prov = {"git_hash": "abc123", "git_dirty": False, "seeds": [42], "python_version": "3.12"}
        r = check_r2_reproducibility(prov)
        assert r.verdict == GateVerdict.pass_

    def test_warn_dirty(self):
        prov = {"git_hash": "abc123", "git_dirty": True, "seeds": [42], "python_version": "3.12"}
        r = check_r2_reproducibility(prov)
        assert r.verdict == GateVerdict.warn

class TestR3MetricIntegrity:
    def test_pass(self, tmp_path):
        art = tmp_path / "artifacts"
        art.mkdir()
        data = np.array([1.0, 2.0])
        np.save(str(art / "x.npy"), data)
        h = hashlib.sha256((art / "x.npy").read_bytes()).hexdigest()
        manifest = {"artifacts": {"x": "artifacts/x.npy"}, "hashes": {"x": f"sha256:{h}"}}
        r = check_r3_metric_integrity(tmp_path, manifest)
        assert r.verdict == GateVerdict.pass_

    def test_fail_corruption(self, tmp_path):
        art = tmp_path / "artifacts"
        art.mkdir()
        np.save(str(art / "x.npy"), np.array([1.0]))
        manifest = {"artifacts": {"x": "artifacts/x.npy"}, "hashes": {"x": "sha256:0000000000"}}
        r = check_r3_metric_integrity(tmp_path, manifest)
        assert r.verdict == GateVerdict.fail

class TestR4BudgetCompliance:
    def test_pass(self):
        manifest = {"metrics": {"runtime_s": 100}, "provenance": {"declared_budget_s": 600}}
        r = check_r4_budget_compliance(manifest)
        assert r.verdict == GateVerdict.pass_

    def test_fail_over_budget(self):
        manifest = {"metrics": {"runtime_s": 1500}, "provenance": {"declared_budget_s": 600}}
        r = check_r4_budget_compliance(manifest)
        assert r.verdict == GateVerdict.fail

class TestRunR1R4:
    def test_returns_four_gates(self, tmp_path):
        manifest = {"version": "0.3.0", "spec_id": "t", "timestamp": "2026",
                     "provenance": {"modality": "ct", "solver": "fbp", "git_hash": "abc", "seeds": [42], "python_version": "3.12"},
                     "metrics": {"runtime_s": 10}, "artifacts": {}}
        results = run_r1_r4(tmp_path, manifest)
        assert set(results.keys()) == {"r1", "r2", "r3", "r4"}

class TestBackwardAlias:
    def test_s1_alias(self):
        assert check_s1_spec_completeness is check_r1_spec_completeness
