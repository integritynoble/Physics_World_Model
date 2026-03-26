"""test_certificate.py -- Certificate model unit tests."""
import pytest
from pwm_core.core.runbundle.certificate import (
    TrustTier, GateVerdict, RiskFlag, GateResult,
    ImagingTriadFlags, DomainFlags, DomainProfileMaturity, Certificate
)

class TestTrustTier:
    def test_all_values(self):
        assert set(t.value for t in TrustTier) == {"rejected", "draft", "author_confirmed", "reproduced", "certified"}

    def test_from_string(self):
        assert TrustTier("draft") == TrustTier.draft
        assert TrustTier("certified") == TrustTier.certified

class TestGateVerdict:
    def test_all_values(self):
        assert set(v.value for v in GateVerdict) == {"pass", "warn", "fail"}

class TestRiskFlag:
    def test_all_values(self):
        assert set(f.value for f in RiskFlag) == {"boundary_risk", "safety_brake", "high_variance", "reviewer_disputed"}

class TestDomainProfileMaturity:
    def test_all_stages(self):
        assert len(DomainProfileMaturity) == 5
        assert DomainProfileMaturity.experimental.value == "experimental"
        assert DomainProfileMaturity.certified_compatible.value == "certified_compatible"

class TestDomainFlags:
    def test_empty(self):
        df = DomainFlags()
        assert df.imaging is None
        assert df.ct_qc is None

    def test_with_imaging(self):
        tf = ImagingTriadFlags(g1_sampling=GateResult(verdict=GateVerdict.pass_))
        df = DomainFlags(imaging=tf)
        assert df.imaging is not None
        assert df.imaging.g1_sampling.verdict == GateVerdict.pass_

class TestCertificate:
    def test_create_minimal(self):
        cert = Certificate(run_id="test_001", spec_id="ct_test", trust_tier=TrustTier.draft)
        assert cert.run_id == "test_001"
        assert cert.trust_tier == TrustTier.draft
        assert cert.risk_flags == []

    def test_is_certified_all_pass(self):
        cert = Certificate(
            run_id="test_002", spec_id="ct_test", trust_tier=TrustTier.draft,
            gate_verdicts={"r1": GateResult(verdict=GateVerdict.pass_), "r2": GateResult(verdict=GateVerdict.pass_)}
        )
        assert cert.is_certified is True

    def test_is_certified_with_failure(self):
        cert = Certificate(
            run_id="test_003", spec_id="ct_test", trust_tier=TrustTier.rejected,
            gate_verdicts={"r1": GateResult(verdict=GateVerdict.fail)}
        )
        assert cert.is_certified is False

    def test_version_stamping_fields(self):
        cert = Certificate(
            run_id="test_004", spec_id="ct_test", trust_tier=TrustTier.draft,
            kernel_version="abc123", profile_version="imaging/v1.0",
            registry_versions={"general": "v1.3"}, product_version="1.0.0"
        )
        assert cert.kernel_version == "abc123"
        assert cert.profile_version == "imaging/v1.0"
        assert cert.registry_versions == {"general": "v1.3"}

    def test_to_dict(self):
        cert = Certificate(run_id="test_005", spec_id="ct_test", trust_tier=TrustTier.draft)
        d = cert.to_dict()
        assert isinstance(d, dict)
        assert d["run_id"] == "test_005"

    def test_backward_compat_triad_flags(self):
        tf = ImagingTriadFlags()
        cert = Certificate(run_id="t", spec_id="s", trust_tier=TrustTier.draft, domain_flags=DomainFlags(imaging=tf))
        assert cert.triad_flags is not None
