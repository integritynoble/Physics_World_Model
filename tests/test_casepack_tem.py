"""Tests for CasePack: Transmission Electron Microscopy (TEM).

Template: tem_graph_v2
Chain: electron_beam_source -> thin_object_phase -> ctf_transfer -> electron_detector_sensor -> gaussian_sensor_noise
"""
import numpy as np
import pytest
import yaml
import os

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import (
    ElectronBeamSource,
    ThinObjectPhase,
    CTFTransfer,
    ElectronDetectorSensor,
)
from pwm_core.physics.electron.tem_helpers import (
    compute_ctf,
    phase_object_transmission,
    ctf_zeros,
)


TEMPLATES_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "packages", "pwm_core", "contrib", "graph_templates.yaml",
)


def _load_template(key):
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data["templates"][key]


class TestCasePackTEM:
    """CasePack acceptance tests for the TEM modality."""

    def test_template_compiles(self):
        """tem_graph_v2 template compiles without error."""
        tpl = _load_template("tem_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "tem_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        assert graph is not None
        assert "source" in graph.node_map
        assert "interaction" in graph.node_map
        assert "ctf" in graph.node_map
        assert "sensor" in graph.node_map
        assert "noise" in graph.node_map

    def test_forward_sanity(self):
        """Mode S: forward pass produces finite, correctly shaped output."""
        tpl = _load_template("tem_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "tem_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)

        rng = np.random.RandomState(42)
        x = rng.rand(16, 16).astype(np.float64) * 0.5
        y = graph.forward(x)

        assert y is not None
        assert np.isfinite(y).all()
        assert y.shape == (16, 16)

    def test_thin_object_transmission(self):
        """phase_object_transmission returns complex array with unit magnitude."""
        x = np.random.RandomState(0).rand(16, 16) * 0.5
        T = phase_object_transmission(x, sigma=0.00729, V=1.0)
        assert T.dtype == np.complex128
        # Magnitude should be exactly 1 (pure phase object)
        np.testing.assert_allclose(np.abs(T), 1.0, atol=1e-12)

    def test_thin_object_phase_primitive(self):
        """ThinObjectPhase primitive produces finite output."""
        top = ThinObjectPhase(params={"sigma": 0.00729})
        x = np.random.RandomState(0).rand(16, 16).astype(np.float64) * 0.5
        y = top.forward(x)
        assert np.isfinite(y).all()
        assert y.shape == (16, 16)

    def test_ctf_transfer_primitive(self):
        """CTFTransfer primitive produces real-valued output of correct shape."""
        ctf_op = CTFTransfer(params={
            "defocus_nm": -50.0,
            "Cs_mm": 1.0,
            "wavelength_pm": 2.5,
        })
        x = np.random.RandomState(0).rand(16, 16).astype(np.float64)
        y = ctf_op.forward(x)
        assert y.shape == (16, 16)
        assert np.isfinite(y).all()
        assert y.dtype == np.float64

    def test_compute_ctf_zeros(self):
        """CTF zero crossings are monotonically increasing."""
        zeros = ctf_zeros(defocus_nm=-50.0, Cs_mm=1.0, wavelength_pm=2.51, n_zeros=5)
        assert len(zeros) >= 3  # at least 3 zeros for reasonable parameters
        # Check monotonically increasing
        for i in range(1, len(zeros)):
            assert zeros[i] > zeros[i - 1], (
                f"CTF zeros not monotonic: {zeros}"
            )

    def test_compute_ctf_shape(self):
        """compute_ctf returns array matching frequency grid shape."""
        N = 32
        fx = np.fft.fftfreq(N)
        fy = np.fft.fftfreq(N)
        ctf = compute_ctf(fx, fy, defocus_nm=-50.0, Cs_mm=1.0, wavelength_pm=2.51)
        assert ctf.shape == (N, N)
        assert np.isfinite(ctf).all()
        # CTF values should be in [-1, 1]
        assert np.all(ctf >= -1.0 - 1e-10)
        assert np.all(ctf <= 1.0 + 1e-10)

    def test_ctf_zero_at_origin(self):
        """CTF is zero at the origin (DC component)."""
        N = 64
        fx = np.fft.fftfreq(N)
        fy = np.fft.fftfreq(N)
        ctf = compute_ctf(fx, fy, defocus_nm=-50.0, Cs_mm=1.0, wavelength_pm=2.51)
        # At q=0, chi=0, so sin(0)=0
        assert abs(ctf[0, 0]) < 1e-10

    def test_thin_object_multi_input(self):
        """ThinObjectPhase.forward_multi combines incident and specimen."""
        top = ThinObjectPhase(params={"sigma": 0.01})
        incident = np.ones((8, 8), dtype=np.float64) * 2.0
        specimen = np.ones((8, 8), dtype=np.float64) * 0.5
        result = top.forward_multi({"incident": incident, "x": specimen})
        # transmission = exp(-0.01 * |0.5|) ~ 0.995
        expected = 2.0 * np.exp(-0.01 * 0.5)
        np.testing.assert_allclose(result, expected, atol=1e-6)
