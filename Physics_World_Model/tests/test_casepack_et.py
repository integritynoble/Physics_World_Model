"""Tests for CasePack: Electron Tomography (ET).

Template: electron_tomography_graph_v2
Chain: electron_beam_source -> thin_object_phase -> electron_detector_sensor -> gaussian_sensor_noise
"""
import numpy as np
import pytest
import yaml
import os

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.physics.electron.et_helpers import (
    tilt_project,
    alignment_shift,
    sirt_recon,
)


TEMPLATES_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "packages", "pwm_core", "contrib", "graph_templates.yaml",
)


def _load_template(key):
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data["templates"][key]


class TestCasePackET:
    """CasePack acceptance tests for the Electron Tomography modality."""

    def test_template_compiles(self):
        """electron_tomography_graph_v2 template compiles without error."""
        tpl = _load_template("electron_tomography_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "et_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        assert graph is not None
        assert "source" in graph.node_map
        assert "interaction" in graph.node_map
        assert "sensor" in graph.node_map
        assert "noise" in graph.node_map

    def test_forward_produces_projection(self):
        """Forward pass produces a 2D output from 2D input."""
        tpl = _load_template("electron_tomography_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "et_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)

        rng = np.random.RandomState(42)
        x = rng.rand(16, 16).astype(np.float64) * 0.5
        y = graph.forward(x)

        assert y is not None
        assert np.isfinite(y).all()
        assert y.shape == (16, 16)

    def test_tilt_project_basic(self):
        """tilt_project produces 2D projection from 3D volume."""
        vol = np.zeros((8, 8, 8), dtype=np.float64)
        vol[3:5, 3:5, 3:5] = 1.0  # small cube in center
        proj = tilt_project(vol, angle_deg=0.0, axis="y")
        assert proj.shape == (8, 8)
        assert np.isfinite(proj).all()
        # Non-trivial: cube should produce non-zero projection
        assert np.sum(proj) > 0

    def test_tilt_project_different_angles(self):
        """Different tilt angles produce different projections."""
        vol = np.zeros((8, 8, 8), dtype=np.float64)
        vol[2:6, 1:3, 4:7] = 1.0  # asymmetric object
        proj_0 = tilt_project(vol, angle_deg=0.0, axis="y")
        proj_45 = tilt_project(vol, angle_deg=45.0, axis="y")
        assert not np.allclose(proj_0, proj_45, atol=1e-3)

    def test_alignment_shift(self):
        """alignment_shift produces shifted 2D projection."""
        rng = np.random.RandomState(0)
        proj = rng.rand(8, 8).astype(np.float64)
        shifted = alignment_shift(proj, dx=1.5, dy=0.5)
        assert shifted.shape == proj.shape
        assert not np.allclose(proj, shifted, atol=1e-3)

    def test_sirt_recon_basic(self):
        """Basic SIRT reconstruction from synthetic tilt series."""
        # Create a simple 3D phantom
        D, H, W = 8, 8, 8
        vol_true = np.zeros((D, H, W), dtype=np.float64)
        vol_true[3:5, 3:5, 3:5] = 1.0

        # Generate tilt series
        angles = np.linspace(-60, 60, 7)
        projections = np.stack(
            [tilt_project(vol_true, a, axis="y") for a in angles],
            axis=0,
        )
        assert projections.shape == (7, H, W)

        # Reconstruct
        vol_recon = sirt_recon(projections, angles, n_iter=5)
        assert vol_recon.shape == (D, H, W)
        assert np.isfinite(vol_recon).all()
        # Reconstruction should have non-zero values where the phantom is
        center_val = vol_recon[3:5, 3:5, 3:5].mean()
        corner_val = vol_recon[0, 0, 0]
        assert center_val > corner_val, (
            f"Center ({center_val:.3f}) should be > corner ({corner_val:.3f})"
        )

    def test_tilt_project_invalid_ndim(self):
        """tilt_project raises ValueError for non-3D input."""
        with pytest.raises(ValueError, match="Expected 3D volume"):
            tilt_project(np.zeros((8, 8)), angle_deg=0.0)

    def test_tilt_project_invalid_axis(self):
        """tilt_project raises ValueError for invalid axis."""
        with pytest.raises(ValueError, match="axis must be"):
            tilt_project(np.zeros((8, 8, 8)), angle_deg=0.0, axis="z")
