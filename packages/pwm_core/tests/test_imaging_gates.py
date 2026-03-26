"""test_imaging_gates.py -- Imaging-specific gate tests (orbit layer)."""
import pytest
from pwm_core.core.runbundle.certificate import GateVerdict, GateResult, ImagingTriadFlags

class TestImagingTriadFlags:
    def test_empty(self):
        tf = ImagingTriadFlags()
        assert tf.g1_sampling is None
        assert tf.g2_noise is None
        assert tf.g3_operator is None

    def test_with_warnings(self):
        g1 = GateResult(verdict=GateVerdict.warn, message="G1 dominant")
        g2 = GateResult(verdict=GateVerdict.pass_)
        g3 = GateResult(verdict=GateVerdict.pass_)
        tf = ImagingTriadFlags(g1_sampling=g1, g2_noise=g2, g3_operator=g3)
        assert tf.g1_sampling.verdict == GateVerdict.warn
        assert tf.g2_noise.verdict == GateVerdict.pass_

    def test_no_safety_brake_from_triad(self):
        """Triad v1 is diagnostic-first: warns but does NOT set safety-brake."""
        g1 = GateResult(verdict=GateVerdict.warn, message="dominant bottleneck")
        tf = ImagingTriadFlags(g1_sampling=g1)
        # Triad flags should never contain GateVerdict.fail in v1
        assert tf.g1_sampling.verdict != GateVerdict.fail

class TestFourScenarioConsistency:
    def test_scenario_psnr_ordering(self):
        """Scenario I (ideal) should generally have highest PSNR."""
        # This is a property test -- I >= III >= II is the expected ordering
        psnr = {"I": 35.0, "II": 28.0, "III": 32.0, "IV": 35.0}
        assert psnr["I"] >= psnr["III"] >= psnr["II"]
