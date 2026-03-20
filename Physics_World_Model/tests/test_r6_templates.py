"""Tests for R6: Physically-correct CT, Photoacoustic, NeRF, 3DGS primitives and templates."""
import numpy as np
import pytest
import yaml

from pwm_core.graph.compiler import GraphCompiler, GraphCompilationError
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import (
    PRIMITIVE_REGISTRY,
    BeerLambert,
    OpticalAbsorption,
    AcousticPropagation,
    VolumeRenderingStub,
    GaussianSplattingStub,
    GaussianSensorNoise,
)


# -----------------------------------------------------------------------
# Primitive unit tests
# -----------------------------------------------------------------------


class TestBeerLambert:
    def test_registration(self):
        assert "beer_lambert" in PRIMITIVE_REGISTRY

    def test_forward_zero_attenuation(self):
        prim = BeerLambert(params={"I_0": 1000.0})
        x = np.zeros((8, 8))  # zero attenuation = full transmission
        y = prim.forward(x)
        np.testing.assert_allclose(y, 1000.0 * np.ones((8, 8)))

    def test_forward_nonzero_attenuation(self):
        prim = BeerLambert(params={"I_0": 1000.0})
        x = np.ones((4, 4)) * np.log(2)  # exp(-ln2) = 0.5
        y = prim.forward(x)
        np.testing.assert_allclose(y, 500.0 * np.ones((4, 4)), rtol=1e-10)

    def test_adjoint_shape(self):
        prim = BeerLambert(params={"I_0": 1000.0})
        x = np.random.randn(8, 8)
        prim.forward(x)  # cache sinogram
        y = np.random.randn(8, 8)
        adj = prim.adjoint(y)
        assert adj.shape == (8, 8)

    def test_nonlinear(self):
        prim = BeerLambert()
        assert not prim.is_linear

    def test_subrole(self):
        assert BeerLambert._physics_subrole == "transduction"


class TestOpticalAbsorption:
    def test_registration(self):
        assert "optical_absorption" in PRIMITIVE_REGISTRY

    def test_forward(self):
        prim = OpticalAbsorption(params={"grueneisen": 0.8, "mu_a": 2.0})
        x = np.ones((8, 8))
        y = prim.forward(x)
        np.testing.assert_allclose(y, 1.6 * np.ones((8, 8)))

    def test_linear(self):
        prim = OpticalAbsorption()
        assert prim.is_linear

    def test_adjoint_matches_forward(self):
        """For a linear scaling operator, adjoint should equal forward."""
        prim = OpticalAbsorption(params={"grueneisen": 0.5, "mu_a": 3.0})
        x = np.random.randn(8, 8)
        y_fwd = prim.forward(x)
        y_adj = prim.adjoint(x)
        np.testing.assert_allclose(y_fwd, y_adj)

    def test_subrole(self):
        assert OpticalAbsorption._physics_subrole == "interaction"

    def test_carrier_type(self):
        assert OpticalAbsorption._carrier_type == "acoustic"


class TestAcousticPropagation:
    def test_registration(self):
        assert "acoustic_propagation" in PRIMITIVE_REGISTRY

    def test_forward_shape(self):
        prim = AcousticPropagation(params={
            "speed_of_sound": 1500.0, "n_sensors": 16
        })
        x = np.random.randn(32, 32)
        y = prim.forward(x)
        assert y.shape[0] == 16  # n_sensors angles

    def test_forward_nonzero(self):
        prim = AcousticPropagation(params={"n_sensors": 8})
        x = np.ones((16, 16))
        y = prim.forward(x)
        assert np.any(y > 0)

    def test_adjoint_shape(self):
        prim = AcousticPropagation(params={
            "n_sensors": 8, "x_shape": [16, 16]
        })
        y = np.random.randn(8, 16)
        x_back = prim.adjoint(y)
        assert x_back.shape == (16, 16)

    def test_linear(self):
        prim = AcousticPropagation()
        assert prim.is_linear

    def test_subrole(self):
        assert AcousticPropagation._physics_subrole == "propagation"


class TestVolumeRenderingStub:
    def test_registration(self):
        assert "volume_rendering_stub" in PRIMITIVE_REGISTRY

    def test_forward_mip_3d(self):
        prim = VolumeRenderingStub(params={"render_mode": "mip"})
        x = np.random.randn(8, 16, 16)
        y = prim.forward(x)
        assert y.shape == (16, 16)
        np.testing.assert_array_equal(y, x.max(axis=0))

    def test_forward_mip_2d_passthrough(self):
        prim = VolumeRenderingStub(params={"render_mode": "mip"})
        x = np.random.randn(16, 16)
        y = prim.forward(x)
        assert y.shape == (16, 16)

    def test_quadrature_raises(self):
        prim = VolumeRenderingStub(params={"render_mode": "quadrature"})
        x = np.random.randn(8, 16, 16)
        with pytest.raises(NotImplementedError, match="PyTorch/JAX"):
            prim.forward(x)

    def test_adjoint_raises(self):
        prim = VolumeRenderingStub()
        with pytest.raises(NotImplementedError):
            prim.adjoint(np.zeros((16, 16)))

    def test_nonlinear(self):
        prim = VolumeRenderingStub()
        assert not prim.is_linear

    def test_tier3(self):
        assert VolumeRenderingStub._physics_tier == "tier3_learned"


class TestGaussianSplattingStub:
    def test_registration(self):
        assert "gaussian_splatting_stub" in PRIMITIVE_REGISTRY

    def test_forward_3d(self):
        prim = GaussianSplattingStub(params={"image_size": [32, 32]})
        x = np.random.randn(8, 16, 16)
        y = prim.forward(x)
        assert y.shape == (32, 32)

    def test_forward_2d(self):
        prim = GaussianSplattingStub(params={"image_size": [64, 64]})
        x = np.random.randn(64, 64)
        y = prim.forward(x)
        assert y.shape == (64, 64)

    def test_adjoint_raises(self):
        prim = GaussianSplattingStub()
        with pytest.raises(NotImplementedError):
            prim.adjoint(np.zeros((64, 64)))

    def test_nonlinear(self):
        prim = GaussianSplattingStub()
        assert not prim.is_linear

    def test_tier3(self):
        assert GaussianSplattingStub._physics_tier == "tier3_learned"


class TestGaussianSensorNoise:
    def test_registration(self):
        assert "gaussian_sensor_noise" in PRIMITIVE_REGISTRY

    def test_forward_adds_noise(self):
        prim = GaussianSensorNoise(params={"sigma": 0.1, "seed": 42})
        x = np.ones((32, 32))
        y = prim.forward(x)
        assert y.shape == (32, 32)
        assert not np.allclose(y, x)

    def test_noise_scale(self):
        """Higher sigma should produce larger deviations."""
        prim_lo = GaussianSensorNoise(params={"sigma": 0.001, "seed": 0})
        prim_hi = GaussianSensorNoise(params={"sigma": 1.0, "seed": 0})
        x = np.ones((64, 64))
        y_lo = prim_lo.forward(x)
        y_hi = prim_hi.forward(x)
        assert np.std(y_hi - x) > np.std(y_lo - x)

    def test_stochastic(self):
        prim = GaussianSensorNoise()
        assert prim.is_stochastic

    def test_not_linear(self):
        prim = GaussianSensorNoise()
        assert not prim.is_linear

    def test_likelihood(self):
        prim = GaussianSensorNoise(params={"sigma": 1.0})
        y = np.array([1.0, 2.0, 3.0])
        y_clean = np.array([1.0, 2.0, 3.0])
        assert prim.likelihood(y, y_clean) == 0.0
        # Non-zero difference
        y_off = np.array([2.0, 2.0, 3.0])
        assert prim.likelihood(y_off, y_clean) > 0.0

    def test_role(self):
        assert GaussianSensorNoise._node_role == "noise"


# -----------------------------------------------------------------------
# Template compilation tests
# -----------------------------------------------------------------------


def _load_templates():
    import os
    templates_path = os.path.join(
        os.path.dirname(__file__), "..",
        "packages", "pwm_core", "contrib", "graph_templates.yaml"
    )
    with open(templates_path) as f:
        data = yaml.safe_load(f)
    return data["templates"]


class TestR6Templates:
    """Test that the corrected v2 templates compile with canonical validation."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.templates = _load_templates()
        self.compiler = GraphCompiler()

    def _compile_template(self, key):
        tpl = dict(self.templates[key])
        tpl.pop("description", None)  # Not part of OperatorGraphSpec
        spec = OperatorGraphSpec.model_validate({
            "graph_id": key,
            **tpl,
        })
        return self.compiler.compile(spec)

    def test_ct_v2_compiles(self):
        graph = self._compile_template("ct_graph_v2")
        assert graph is not None
        # Should have beer_lambert node
        assert "beer_lambert" in graph.node_map

    def test_ct_v2_forward(self):
        graph = self._compile_template("ct_graph_v2")
        x = np.random.RandomState(42).rand(64, 64) * 0.5
        y = graph.forward(x)
        assert y.shape[0] > 0
        # Beer-Lambert ensures output is positive (photon counts)
        # After noise, values should still be mostly positive
        assert np.mean(y > 0) > 0.5

    def test_ct_beer_lambert_physics(self):
        """Beer-Lambert produces correct exponential attenuation."""
        bl = BeerLambert(params={"I_0": 1000.0})
        sinogram = np.array([[0.0, 0.5, 1.0, 2.0]])
        transmitted = bl.forward(sinogram)
        expected = 1000.0 * np.exp(-sinogram)
        np.testing.assert_allclose(transmitted, expected)

    def test_photoacoustic_v2_compiles(self):
        graph = self._compile_template("photoacoustic_graph_v2")
        assert graph is not None
        assert "absorption" in graph.node_map
        assert "propagate" in graph.node_map

    def test_photoacoustic_v2_has_carrier_transition(self):
        """PA template declares photon→acoustic carrier transition."""
        tpl = self.templates["photoacoustic_graph_v2"]
        transitions = tpl["metadata"].get("carrier_transitions", [])
        assert "photon->acoustic" in transitions

    def test_photoacoustic_v2_gaussian_noise(self):
        """PA template uses Gaussian noise (not Poisson)."""
        tpl = self.templates["photoacoustic_graph_v2"]
        noise_node = [n for n in tpl["nodes"] if n.get("role") == "noise"][0]
        assert noise_node["primitive_id"] == "gaussian_sensor_noise"

    def test_nerf_v2_compiles(self):
        graph = self._compile_template("nerf_graph_v2")
        assert graph is not None
        assert "volume_render" in graph.node_map

    def test_nerf_v2_tier3_metadata(self):
        tpl = self.templates["nerf_graph_v2"]
        assert tpl["metadata"].get("physics_tier") == "tier3_learned"

    def test_nerf_v2_forward(self):
        graph = self._compile_template("nerf_graph_v2")
        x = np.random.RandomState(42).rand(16, 64, 64)
        y = graph.forward(x)
        assert y.shape == (64, 64)

    def test_3dgs_v2_compiles(self):
        graph = self._compile_template("gaussian_splatting_graph_v2")
        assert graph is not None
        assert "splat" in graph.node_map

    def test_3dgs_v2_tier3_metadata(self):
        tpl = self.templates["gaussian_splatting_graph_v2"]
        assert tpl["metadata"].get("physics_tier") == "tier3_learned"

    def test_3dgs_v2_forward(self):
        graph = self._compile_template("gaussian_splatting_graph_v2")
        x = np.random.RandomState(42).rand(16, 64, 64)
        y = graph.forward(x)
        assert y.shape == (64, 64)

    def test_all_26_v2_templates_compile(self):
        """All 26 v2 templates still compile (no regression)."""
        v2_keys = [k for k in self.templates if k.endswith("_v2")]
        assert len(v2_keys) >= 26, f"Expected >=26 v2 templates, got {len(v2_keys)}"
        for key in v2_keys:
            graph = self._compile_template(key)
            assert graph is not None, f"Template {key} failed to compile"
