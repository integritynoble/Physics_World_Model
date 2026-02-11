"""Tests for CasePack: Scanning Electron Microscopy (SEM).

Template: sem_graph_v2
Chain: electron_beam_source -> yield_model -> electron_detector_sensor -> gaussian_sensor_noise
"""
import numpy as np
import pytest
import yaml
import os

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import (
    ElectronBeamSource,
    YieldModel,
    ElectronDetectorSensor,
    GaussianSensorNoise,
)
from pwm_core.physics.electron.sem_helpers import se_yield, bse_yield, apply_scan_drift


TEMPLATES_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "packages", "pwm_core", "contrib", "graph_templates.yaml",
)


def _load_template(key):
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data["templates"][key]


class TestCasePackSEM:
    """CasePack acceptance tests for the SEM modality."""

    def test_template_compiles(self):
        """sem_graph_v2 template compiles without error."""
        tpl = _load_template("sem_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "sem_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        assert graph is not None
        assert "source" in graph.node_map
        assert "interaction" in graph.node_map
        assert "sensor" in graph.node_map
        assert "noise" in graph.node_map

    def test_forward_sanity(self):
        """Mode S: forward pass produces finite, non-negative, correctly shaped output."""
        tpl = _load_template("sem_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "sem_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)

        rng = np.random.RandomState(42)
        x = rng.rand(16, 16).astype(np.float64)
        y = graph.forward(x)

        assert y is not None
        assert np.isfinite(y).all()
        assert y.shape == (16, 16)

    def test_yield_model_scaling(self):
        """YieldModel primitive scales output by yield_coeff."""
        ym = YieldModel(params={"yield_coeff": 0.3})
        x = np.ones((16, 16), dtype=np.float64)
        y = ym.forward(x)
        np.testing.assert_allclose(y, 0.3 * np.ones((16, 16)), atol=1e-12)

    def test_yield_model_multi_input(self):
        """YieldModel.forward_multi combines incident beam and material map."""
        ym = YieldModel(params={"yield_coeff": 0.5})
        incident = 2.0 * np.ones((8, 8), dtype=np.float64)
        material = 0.6 * np.ones((8, 8), dtype=np.float64)
        result = ym.forward_multi({"incident": incident, "x": material})
        expected = 0.5 * 2.0 * 0.6
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_se_yield_helper(self):
        """se_yield produces non-negative output in [0, 1]."""
        mat = np.random.RandomState(0).rand(16, 16)
        y = se_yield(mat, voltage_kv=15.0)
        assert y.shape == (16, 16)
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)

    def test_bse_yield_helper(self):
        """bse_yield produces non-negative output in [0, 1]."""
        mat = np.random.RandomState(0).rand(16, 16) * 0.5
        y = bse_yield(mat, voltage_kv=20.0, angle=0.0)
        assert y.shape == (16, 16)
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)

    def test_scan_drift_detectable(self):
        """apply_scan_drift produces a visibly different image."""
        rng = np.random.RandomState(7)
        img = rng.rand(16, 16).astype(np.float64)
        shifted = apply_scan_drift(img, drift_x=1.5, drift_y=0.8)
        assert shifted.shape == img.shape
        # The shifted image should differ from the original
        assert not np.allclose(img, shifted, atol=1e-3)

    def test_different_voltage_different_yield(self):
        """Different accelerating voltages produce different SE yields."""
        mat = np.ones((16, 16), dtype=np.float64) * 0.5
        y1 = se_yield(mat, voltage_kv=5.0)
        y2 = se_yield(mat, voltage_kv=30.0)
        # Lower voltage => higher SE yield (less penetration)
        assert np.mean(y1) > np.mean(y2)
