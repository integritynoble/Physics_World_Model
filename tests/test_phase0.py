"""Tests for Phase 0: IR hardening, validation, registries."""
import numpy as np
import pytest
import yaml
import os

# -----------------------------------------------------------------------
# Test: Modality Registry (P0.3)
# -----------------------------------------------------------------------


class TestModalityRegistry:
    def test_load_modalities(self):
        from pwm_core.core.modality_registry import load_modalities, clear_cache
        clear_cache()
        mods = load_modalities()
        assert len(mods) >= 25
        assert "widefield" in mods
        assert "ct" in mods

    def test_get_modality(self):
        from pwm_core.core.modality_registry import get_modality, clear_cache
        clear_cache()
        mod = get_modality("widefield")
        assert mod.display_name != ""
        assert mod.category == "microscopy"

    def test_get_unknown_raises(self):
        from pwm_core.core.modality_registry import get_modality
        with pytest.raises(KeyError, match="Unknown modality"):
            get_modality("nonexistent_modality_xyz")

    def test_modality_info_fields(self):
        from pwm_core.core.modality_registry import get_modality, clear_cache
        clear_cache()
        mod = get_modality("ct")
        assert hasattr(mod, "requires_x_interaction")
        assert hasattr(mod, "acceptance_tier")


# -----------------------------------------------------------------------
# Test: MetricRegistry (P0.5)
# -----------------------------------------------------------------------


class TestMetricRegistry:
    def test_psnr_perfect(self):
        from pwm_core.core.metric_registry import PSNR
        x = np.ones((32, 32))
        assert PSNR()(x, x) == float('inf')

    def test_psnr_nonzero(self):
        from pwm_core.core.metric_registry import PSNR
        x = np.ones((32, 32))
        y = x + 0.1 * np.random.randn(32, 32)
        psnr = PSNR()(y, x)
        assert psnr > 0 and psnr < 100

    def test_ssim_perfect(self):
        from pwm_core.core.metric_registry import SSIM
        x = np.random.rand(32, 32)
        val = SSIM()(x, x)
        assert abs(val - 1.0) < 0.01

    def test_crc_perfect(self):
        from pwm_core.core.metric_registry import CRC
        rng = np.random.RandomState(42)
        x = rng.rand(32, 32)  # uniform background ~0.5
        x[10:20, 10:20] += 5.0  # hot region well above p90
        crc = CRC()(x, x)
        assert abs(crc - 1.0) < 0.1

    def test_cnr_positive(self):
        from pwm_core.core.metric_registry import CNR
        rng = np.random.RandomState(42)
        x = rng.rand(32, 32)  # uniform background ~0.5
        x[10:20, 10:20] += 5.0  # hot region well above p80
        cnr = CNR()(x, x)
        assert cnr > 0

    def test_frc_perfect(self):
        from pwm_core.core.metric_registry import FRC
        x = np.random.rand(32, 32)
        frc = FRC()(x, x)
        assert abs(frc - 1.0) < 0.01

    def test_spectral_angle_perfect(self):
        from pwm_core.core.metric_registry import SpectralAngle
        x = np.random.rand(16, 16, 8)
        sa = SpectralAngle()(x, x)
        assert abs(sa - 1.0) < 0.01

    def test_build_metric(self):
        from pwm_core.core.metric_registry import build_metric
        m = build_metric("psnr")
        assert m.name == "psnr"

    def test_build_metric_unknown(self):
        from pwm_core.core.metric_registry import build_metric
        with pytest.raises(KeyError):
            build_metric("nonexistent_metric")

    def test_registry_has_all(self):
        from pwm_core.core.metric_registry import METRIC_REGISTRY
        expected = {"psnr", "ssim", "nll", "crc", "cnr", "frc", "spectral_angle"}
        assert expected.issubset(set(METRIC_REGISTRY.keys()))


# -----------------------------------------------------------------------
# Test: CasePack Runner (P0.4)
# -----------------------------------------------------------------------


class TestCasePackRunner:
    def test_run_unknown_modality(self):
        from pwm_core.core.casepack_runner import run_casepack
        result = run_casepack("nonexistent_xyz", quick=True)
        assert result.modality == "nonexistent_xyz"
        assert not result.forward_ok

    def test_run_widefield(self):
        from pwm_core.core.casepack_runner import run_casepack
        result = run_casepack("widefield", quick=True)
        assert result.modality == "widefield"
        # May or may not have a template -- check gracefully
        # At minimum it should not crash

    def test_result_fields(self):
        from pwm_core.core.casepack_runner import CasePackResult
        r = CasePackResult(modality="test")
        assert r.forward_ok is False
        assert r.recon_psnr is None
        assert r.primary_metric is None


# -----------------------------------------------------------------------
# Test: Shape Validation (P0.1)
# -----------------------------------------------------------------------


class TestShapeValidation:
    def test_compatible_shapes_pass(self):
        """Nodes with compatible output_shape/input_shape should compile."""
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.graph.graph_spec import OperatorGraphSpec
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_compat",
            "nodes": [
                {"node_id": "a", "primitive_id": "identity", "params": {"output_shape": [64, 64]}},
                {"node_id": "b", "primitive_id": "identity", "params": {"input_shape": [64, 64]}},
            ],
            "edges": [{"source": "a", "target": "b"}],
        })
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        assert graph is not None

    def test_incompatible_shapes_raise(self):
        from pwm_core.graph.compiler import GraphCompiler, GraphCompilationError
        from pwm_core.graph.graph_spec import OperatorGraphSpec
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_incompat",
            "nodes": [
                {"node_id": "a", "primitive_id": "identity", "params": {"output_shape": [64, 64]}},
                {"node_id": "b", "primitive_id": "identity", "params": {"input_shape": [32, 32]}},
            ],
            "edges": [{"source": "a", "target": "b"}],
        })
        compiler = GraphCompiler()
        with pytest.raises(GraphCompilationError, match="incompatible"):
            compiler.compile(spec)

    def test_dynamic_axes_tolerated(self):
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.graph.graph_spec import OperatorGraphSpec
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_dynamic",
            "nodes": [
                {"node_id": "a", "primitive_id": "identity", "params": {"output_shape": [-1, 64]}},
                {"node_id": "b", "primitive_id": "identity", "params": {"input_shape": [32, 64]}},
            ],
            "edges": [{"source": "a", "target": "b"}],
        })
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        assert graph is not None

    def test_existing_v2_templates_still_compile(self):
        """All existing v2 templates must still compile after shape validation."""
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.graph.graph_spec import OperatorGraphSpec
        templates_path = os.path.join(
            os.path.dirname(__file__), "..",
            "packages", "pwm_core", "contrib", "graph_templates.yaml"
        )
        if not os.path.exists(templates_path):
            pytest.skip("Templates not found")
        with open(templates_path) as f:
            data = yaml.safe_load(f)
        templates = data.get("templates", {})
        compiler = GraphCompiler()
        v2_keys = [k for k in templates if k.endswith("_v2")]
        assert len(v2_keys) >= 26
        for key in v2_keys:
            tpl = dict(templates[key])
            tpl.pop("description", None)
            spec = OperatorGraphSpec.model_validate({"graph_id": key, **tpl})
            graph = compiler.compile(spec)
            assert graph is not None


# -----------------------------------------------------------------------
# Test: Source-to-x wiring (P0.2, D2)
# -----------------------------------------------------------------------


class TestSourceToXWiring:
    def test_requires_x_interaction_passes_with_interaction_node(self):
        """Graph with interaction subrole passes when requires_x_interaction=True."""
        from pwm_core.graph.canonical import validate_canonical_chain
        from pwm_core.graph.graph_spec import OperatorGraphSpec
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_d2_pass",
            "metadata": {
                "canonical_chain": True,
                "requires_x_interaction": True,
            },
            "nodes": [
                {"node_id": "src", "primitive_id": "photon_source", "params": {}},
                {"node_id": "interact", "primitive_id": "optical_absorption", "params": {}},
                {"node_id": "sens", "primitive_id": "photon_sensor", "params": {}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor", "params": {}},
            ],
            "edges": [
                {"source": "src", "target": "interact"},
                {"source": "interact", "target": "sens"},
                {"source": "sens", "target": "noise"},
            ],
        })
        # Should not raise
        validate_canonical_chain(spec)

    def test_requires_x_interaction_fails_without_interaction_node(self):
        """Graph without interaction/transduction fails when requires_x_interaction=True."""
        from pwm_core.graph.canonical import validate_canonical_chain
        from pwm_core.graph.compiler import GraphCompilationError
        from pwm_core.graph.graph_spec import OperatorGraphSpec
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_d2_fail",
            "metadata": {
                "canonical_chain": True,
                "requires_x_interaction": True,
            },
            "nodes": [
                {"node_id": "src", "primitive_id": "photon_source", "params": {}},
                {"node_id": "blur", "primitive_id": "conv2d", "params": {}},
                {"node_id": "sens", "primitive_id": "photon_sensor", "params": {}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor", "params": {}},
            ],
            "edges": [
                {"source": "src", "target": "blur"},
                {"source": "blur", "target": "sens"},
                {"source": "sens", "target": "noise"},
            ],
        })
        with pytest.raises(GraphCompilationError, match="requires x-interaction"):
            validate_canonical_chain(spec)

    def test_no_requires_x_passes_without_interaction(self):
        """Without requires_x_interaction, no interaction node needed."""
        from pwm_core.graph.canonical import validate_canonical_chain
        from pwm_core.graph.graph_spec import OperatorGraphSpec
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_d2_optional",
            "metadata": {
                "canonical_chain": True,
                "requires_x_interaction": False,
            },
            "nodes": [
                {"node_id": "src", "primitive_id": "photon_source", "params": {}},
                {"node_id": "blur", "primitive_id": "conv2d", "params": {}},
                {"node_id": "sens", "primitive_id": "photon_sensor", "params": {}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor", "params": {}},
            ],
            "edges": [
                {"source": "src", "target": "blur"},
                {"source": "blur", "target": "sens"},
                {"source": "sens", "target": "noise"},
            ],
        })
        validate_canonical_chain(spec)


# -----------------------------------------------------------------------
# Test: Optics Convention (P0.6)
# -----------------------------------------------------------------------


class TestOpticsConvention:
    def test_photon_field_spec_creation(self):
        from pwm_core.graph.optics_convention import PhotonFieldSpec
        pfs = PhotonFieldSpec(wavelength_m=532e-9, grid_shape=(64, 64), pixel_pitch_m=6.5e-6)
        assert pfs.grid_shape == (64, 64)
        assert pfs.max_freq == pytest.approx(1 / (2 * 6.5e-6))

    def test_freq_grid_shape(self):
        from pwm_core.graph.optics_convention import PhotonFieldSpec
        pfs = PhotonFieldSpec(grid_shape=(32, 32), pixel_pitch_m=1e-5)
        fy, fx = pfs.freq_grid()
        assert fy.shape == (32, 32)
        assert fx.shape == (32, 32)

    def test_real_grid_shape(self):
        from pwm_core.graph.optics_convention import PhotonFieldSpec
        pfs = PhotonFieldSpec(grid_shape=(32, 32), pixel_pitch_m=1e-5)
        y, x = pfs.real_grid()
        assert y.shape == (32, 32)

    def test_validate_output_shape(self):
        from pwm_core.graph.optics_convention import PhotonFieldSpec
        pfs = PhotonFieldSpec(grid_shape=(64, 64))
        good = np.zeros((64, 64))
        bad = np.zeros((32, 32))
        assert pfs.validate_output_shape(good)
        assert not pfs.validate_output_shape(bad)

    def test_freq_pitch(self):
        from pwm_core.graph.optics_convention import PhotonFieldSpec
        pfs = PhotonFieldSpec(grid_shape=(64, 64), pixel_pitch_m=1e-5)
        fy, fx = pfs.freq_pitch
        assert fy == pytest.approx(1 / (64 * 1e-5))
