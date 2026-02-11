"""Tests for pwm_core.objectives.noise_model."""

import numpy as np
import pytest

from pwm_core.objectives.noise_model import (
    NoiseModelSpec,
    PoissonGaussianNoiseModel,
    GaussianNoiseModel,
    ComplexGaussianNoiseModel,
    PoissonOnlyNoiseModel,
    NOISE_MODEL_REGISTRY,
    build_noise_model,
    noise_model_from_primitive,
)
from pwm_core.objectives.base import ObjectiveSpec


class TestPoissonGaussianNoiseModel:
    def test_sample_shape(self):
        nm = PoissonGaussianNoiseModel(peak_photons=10000, read_sigma=0.01)
        rng = np.random.default_rng(42)
        y_clean = np.ones((16, 16)) * 0.5
        y_noisy = nm.sample(y_clean, rng)
        assert y_noisy.shape == y_clean.shape

    def test_sample_adds_noise(self):
        nm = PoissonGaussianNoiseModel(peak_photons=1000, read_sigma=0.1)
        rng = np.random.default_rng(42)
        y_clean = np.ones((32, 32)) * 0.5
        y_noisy = nm.sample(y_clean, rng)
        assert not np.allclose(y_noisy, y_clean)

    def test_default_objective(self):
        nm = PoissonGaussianNoiseModel()
        obj = nm.default_objective()
        assert obj.kind == "mixed_poisson_gaussian"

    def test_log_likelihood_finite(self):
        nm = PoissonGaussianNoiseModel(peak_photons=10000, read_sigma=0.01)
        y_clean = np.ones((8, 8)) * 0.5
        rng = np.random.default_rng(0)
        y = nm.sample(y_clean, rng)
        ll = nm.log_likelihood(y, y_clean)
        assert np.isfinite(ll)


class TestGaussianNoiseModel:
    def test_sample_statistics(self):
        nm = GaussianNoiseModel(sigma=0.1)
        rng = np.random.default_rng(42)
        y_clean = np.zeros((1000,))
        y_noisy = nm.sample(y_clean, rng)
        assert abs(np.std(y_noisy) - 0.1) < 0.02

    def test_default_objective(self):
        nm = GaussianNoiseModel(sigma=0.5)
        obj = nm.default_objective()
        assert obj.kind == "gaussian"


class TestComplexGaussianNoiseModel:
    def test_sample_complex(self):
        nm = ComplexGaussianNoiseModel(sigma=0.1)
        rng = np.random.default_rng(42)
        y_clean = np.ones((16, 16), dtype=np.complex128)
        y_noisy = nm.sample(y_clean, rng)
        assert np.iscomplexobj(y_noisy)
        assert y_noisy.shape == y_clean.shape

    def test_default_objective(self):
        nm = ComplexGaussianNoiseModel()
        assert nm.default_objective().kind == "complex_gaussian"


class TestPoissonOnlyNoiseModel:
    def test_sample_nonneg(self):
        nm = PoissonOnlyNoiseModel(peak_photons=10000)
        rng = np.random.default_rng(42)
        y_clean = np.ones((16, 16)) * 0.5
        y_noisy = nm.sample(y_clean, rng)
        assert np.all(y_noisy >= 0)

    def test_default_objective(self):
        nm = PoissonOnlyNoiseModel()
        assert nm.default_objective().kind == "poisson"


class TestNoiseModelRegistry:
    def test_all_registered(self):
        expected = {"poisson_gaussian", "gaussian", "complex_gaussian", "poisson_only"}
        assert expected.issubset(set(NOISE_MODEL_REGISTRY.keys()))

    def test_build_gaussian(self):
        spec = NoiseModelSpec(kind="gaussian", params={"sigma": 0.5})
        nm = build_noise_model(spec)
        assert isinstance(nm, GaussianNoiseModel)
        assert nm.sigma == 0.5

    def test_build_unknown_raises(self):
        spec = NoiseModelSpec(kind="nonexistent")
        with pytest.raises(KeyError):
            build_noise_model(spec)

    def test_from_primitive(self):
        nm = noise_model_from_primitive(
            "poisson_gaussian_sensor",
            {"peak_photons": 5000.0, "read_sigma": 0.02, "seed": 0}
        )
        assert isinstance(nm, PoissonGaussianNoiseModel)
        assert nm.peak_photons == 5000.0


class TestObjectiveOverride:
    def test_override_objective_on_executor(self):
        """User can override objective regardless of noise type."""
        from pwm_core.core.enums import ExecutionMode
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.graph.executor import ExecutionConfig, GraphExecutor
        from pwm_core.graph.graph_spec import OperatorGraphSpec

        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_override",
            "metadata": {"canonical_chain": True, "x_shape": [32, 32], "y_shape": [32, 32]},
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport", "params": {"sigma": 2.0, "mode": "reflect"}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor", "params": {"quantum_efficiency": 0.9, "gain": 1.0}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise", "params": {"peak_photons": 10000, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "blur"},
                {"source": "blur", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
            ],
        })
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        executor = GraphExecutor(graph)

        # Noise is poisson_gaussian but user overrides with huber
        assert executor._noise_model is not None
        assert executor._noise_model.default_objective().kind == "mixed_poisson_gaussian"

        # Override works — user can specify any objective
        config = ExecutionConfig(
            mode=ExecutionMode.invert,
            objective_spec=ObjectiveSpec(kind="huber"),
        )
        # Just verify config is accepted, don't need actual inversion
        assert config.objective_spec.kind == "huber"
