"""test_triad_report.py -- ImagingTriadReport + DomainDiagnosticReport tests."""
import pytest
from pwm_core.core.runbundle.triad_report import (
    DomainDiagnosticReport, ImagingTriadReport, CTQCDiagnosticReport,
    GateAttribution, TriadReport  # backward alias
)

class TestGateAttribution:
    def test_create(self):
        ga = GateAttribution(gate="g1", confidence=0.8, verdict="warn", evidence={"oracle_gap": 3.5})
        assert ga.confidence == 0.8
        assert ga.gate == "g1"

class TestImagingTriadReport:
    def test_is_complete_true(self):
        ga = lambda g: GateAttribution(gate=g, confidence=0.5, verdict="pass")
        r = ImagingTriadReport(run_id="t", spec_id="s", g1_sampling=ga("g1"), g2_noise=ga("g2"), g3_operator=ga("g3"))
        assert r.is_complete is True

    def test_is_complete_false(self):
        ga0 = GateAttribution(gate="g1", confidence=0.0, verdict="pass")
        ga1 = GateAttribution(gate="g2", confidence=0.5, verdict="pass")
        ga2 = GateAttribution(gate="g3", confidence=0.5, verdict="pass")
        r = ImagingTriadReport(run_id="t", spec_id="s", g1_sampling=ga0, g2_noise=ga1, g3_operator=ga2)
        assert r.is_complete is False

    def test_domain_is_imaging(self):
        ga = lambda g: GateAttribution(gate=g, confidence=0.5, verdict="pass")
        r = ImagingTriadReport(run_id="t", spec_id="s", g1_sampling=ga("g1"), g2_noise=ga("g2"), g3_operator=ga("g3"))
        assert r.domain == "imaging"

class TestCTQCDiagnosticReport:
    def test_is_complete(self):
        r = CTQCDiagnosticReport(run_id="t", spec_id="s", noise_std_hu=5.0, cnr=12.0)
        assert r.is_complete is True

    def test_dominant_concern_drift(self):
        r = CTQCDiagnosticReport(run_id="t", spec_id="s", drift_detected=True, noise_std_hu=5.0, cnr=12.0)
        assert r.dominant_concern == "drift"

    def test_domain_is_ct_qc(self):
        r = CTQCDiagnosticReport(run_id="t", spec_id="s")
        assert r.domain == "ct_qc"

class TestBackwardAlias:
    def test_triad_report_alias(self):
        assert TriadReport is ImagingTriadReport
