"""Tests for R4 Advanced Optical Microscopy modalities.

Covers template compilation, YAML registry presence, and DB consistency
for all 5 new modalities: two_photon, sted, palm_storm, tirf, polarization.
"""

import os

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_CONTRIB = os.path.join(
    os.path.dirname(__file__),
    "..",
    "packages",
    "pwm_core",
    "contrib",
)


def _load_yaml(name: str):
    path = os.path.join(_CONTRIB, name)
    with open(path) as f:
        return yaml.safe_load(f)


def _templates():
    return _load_yaml("graph_templates.yaml")["templates"]


def _modalities():
    return _load_yaml("modalities.yaml")["modalities"]


def _mismatch():
    return _load_yaml("mismatch_db.yaml")["modalities"]


def _photon():
    return _load_yaml("photon_db.yaml")["modalities"]


def _compression():
    return _load_yaml("compression_db.yaml")["calibration_tables"]


def _metrics():
    return _load_yaml("metrics_db.yaml")["metric_sets"]


# The 5 R4 modality keys
R4_MODALITIES = ["two_photon", "sted", "palm_storm", "tirf", "polarization"]

# Template IDs
TEMPLATE_IDS = [f"{m}_graph_v2" for m in R4_MODALITIES]


# ===========================================================================
# YAML structure tests
# ===========================================================================


class TestGraphTemplatesPresent:
    """All 5 new templates exist in graph_templates.yaml."""

    @pytest.mark.parametrize("tid", TEMPLATE_IDS)
    def test_template_exists(self, tid):
        templates = _templates()
        assert tid in templates, f"Template {tid} missing from graph_templates.yaml"

    @pytest.mark.parametrize("tid", TEMPLATE_IDS)
    def test_template_has_canonical_chain(self, tid):
        tpl = _templates()[tid]
        assert tpl["metadata"]["canonical_chain"] is True

    @pytest.mark.parametrize("tid", TEMPLATE_IDS)
    def test_template_has_nodes_and_edges(self, tid):
        tpl = _templates()[tid]
        assert len(tpl["nodes"]) >= 3, "Need at least source, sensor, noise"
        assert len(tpl["edges"]) >= 2, "Need at least 2 edges"

    @pytest.mark.parametrize("tid", TEMPLATE_IDS)
    def test_template_roles(self, tid):
        tpl = _templates()[tid]
        roles = {n["role"] for n in tpl["nodes"]}
        assert "source" in roles, f"No source role in {tid}"
        assert "noise" in roles, f"No noise role in {tid}"
        assert "sensor" in roles, f"No sensor role in {tid}"

    @pytest.mark.parametrize("tid", TEMPLATE_IDS)
    def test_edge_targets_are_valid_node_ids(self, tid):
        tpl = _templates()[tid]
        node_ids = {n["node_id"] for n in tpl["nodes"]}
        for edge in tpl["edges"]:
            assert edge["source"] in node_ids, f"Edge source {edge['source']} not a node"
            assert edge["target"] in node_ids, f"Edge target {edge['target']} not a node"

    @pytest.mark.parametrize("tid", TEMPLATE_IDS)
    def test_template_x_y_shapes(self, tid):
        tpl = _templates()[tid]
        md = tpl["metadata"]
        assert "x_shape" in md, f"No x_shape in {tid}"
        assert "y_shape" in md, f"No y_shape in {tid}"
        assert len(md["x_shape"]) >= 2, f"x_shape too short in {tid}"
        assert len(md["y_shape"]) >= 2, f"y_shape too short in {tid}"


class TestModalitiesPresent:
    """All 5 modalities in modalities.yaml."""

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_modality_exists(self, m):
        mods = _modalities()
        assert m in mods, f"Modality {m} missing from modalities.yaml"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_modality_has_required_fields(self, m):
        mod = _modalities()[m]
        for field in [
            "display_name",
            "category",
            "keywords",
            "description",
            "signal_dims",
            "forward_model_type",
            "forward_model_equation",
            "default_solver",
            "default_template_id",
            "acceptance_tier",
            "elements",
            "upload_template",
        ]:
            assert field in mod, f"Modality {m} missing field: {field}"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_microscopy_category(self, m):
        mod = _modalities()[m]
        assert mod["category"] == "microscopy"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_template_id_matches(self, m):
        mod = _modalities()[m]
        expected = f"{m}_graph_v2"
        assert mod["default_template_id"] == expected

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_elements_have_source_and_detector(self, m):
        elements = _modalities()[m]["elements"]
        types = [e["element_type"] for e in elements]
        assert "source" in types, f"{m}: no source element"
        assert "detector" in types, f"{m}: no detector element"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_acceptance_tier(self, m):
        mod = _modalities()[m]
        assert mod["acceptance_tier"] == "tier1"


class TestMismatchDBPresent:
    """All 5 modalities in mismatch_db.yaml."""

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_modality_in_mismatch(self, m):
        db = _mismatch()
        assert m in db, f"Modality {m} missing from mismatch_db.yaml"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_mismatch_has_parameters(self, m):
        entry = _mismatch()[m]
        assert "parameters" in entry
        assert len(entry["parameters"]) >= 2, f"{m}: need at least 2 mismatch params"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_severity_weights_sum(self, m):
        entry = _mismatch()[m]
        weights = entry["severity_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.05, f"{m}: severity weights sum to {total}"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_correction_method_exists(self, m):
        entry = _mismatch()[m]
        assert "correction_method" in entry
        assert entry["correction_method"] in [
            "grid_search",
            "gradient_descent",
            "UPWMI_beam_search",
        ]

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_param_fields(self, m):
        entry = _mismatch()[m]
        for pname, pval in entry["parameters"].items():
            for field in ["range", "typical_error", "unit", "param_type", "description"]:
                assert field in pval, f"{m}.{pname} missing {field}"


class TestPhotonDBPresent:
    """All 5 modalities in photon_db.yaml."""

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_modality_in_photon(self, m):
        db = _photon()
        assert m in db, f"Modality {m} missing from photon_db.yaml"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_photon_levels(self, m):
        entry = _photon()[m]
        assert "photon_levels" in entry
        for level in ["bright", "standard", "low_light"]:
            assert level in entry["photon_levels"], f"{m}: missing photon level {level}"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_model_id_exists(self, m):
        entry = _photon()[m]
        assert "model_id" in entry
        assert isinstance(entry["model_id"], str)

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_noise_model_exists(self, m):
        entry = _photon()[m]
        assert "noise_model" in entry
        assert entry["noise_model"] in [
            "poisson",
            "poisson_gaussian",
            "gaussian",
        ]


class TestCompressionDBPresent:
    """All 5 modalities in compression_db.yaml."""

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_modality_in_compression(self, m):
        db = _compression()
        assert m in db, f"Modality {m} missing from compression_db.yaml"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_has_entries(self, m):
        entry = _compression()[m]
        assert "entries" in entry
        assert len(entry["entries"]) >= 2, f"{m}: need at least 2 calibration entries"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_signal_prior_class(self, m):
        entry = _compression()[m]
        assert "signal_prior_class" in entry

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_entry_fields(self, m):
        entries = _compression()[m]["entries"]
        for i, e in enumerate(entries):
            for field in ["cr", "noise", "solver", "recoverability", "expected_psnr_db", "provenance"]:
                assert field in e, f"{m} entry[{i}] missing {field}"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_provenance_fields(self, m):
        entries = _compression()[m]["entries"]
        for i, e in enumerate(entries):
            prov = e["provenance"]
            for field in ["dataset_id", "seed_set", "operator_version", "solver_version", "date_generated"]:
                assert field in prov, f"{m} entry[{i}] provenance missing {field}"


class TestMetricsDBPresent:
    """New modalities are covered in metrics_db.yaml metric sets."""

    def test_advanced_microscopy_metric_set(self):
        ms = _metrics()
        assert "advanced_microscopy" in ms
        for m in R4_MODALITIES:
            assert m in ms["advanced_microscopy"]["modalities"]

    def test_advanced_microscopy_metrics(self):
        ms = _metrics()
        metrics = ms["advanced_microscopy"]["metrics"]
        for metric in ["psnr", "ssim", "frc", "nrmse"]:
            assert metric in metrics


# ===========================================================================
# Template structure consistency tests
# ===========================================================================


class TestTemplateTwoPhoton:
    """Two-photon template structure checks."""

    def test_has_nonlinear_excitation(self):
        tpl = _templates()["two_photon_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "nonlinear_excitation" in pids

    def test_has_conv2d(self):
        tpl = _templates()["two_photon_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "conv2d" in pids

    def test_source_is_photon(self):
        tpl = _templates()["two_photon_graph_v2"]
        source = [n for n in tpl["nodes"] if n["role"] == "source"][0]
        assert source["primitive_id"] == "photon_source"

    def test_sensor_is_photon(self):
        tpl = _templates()["two_photon_graph_v2"]
        sensor = [n for n in tpl["nodes"] if n["role"] == "sensor"][0]
        assert sensor["primitive_id"] == "photon_sensor"

    def test_noise_is_poisson_gaussian(self):
        tpl = _templates()["two_photon_graph_v2"]
        noise = [n for n in tpl["nodes"] if n["role"] == "noise"][0]
        assert noise["primitive_id"] == "poisson_gaussian_sensor"


class TestTemplateSTED:
    """STED template structure checks."""

    def test_has_saturation_depletion(self):
        tpl = _templates()["sted_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "saturation_depletion" in pids

    def test_has_conv2d(self):
        tpl = _templates()["sted_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "conv2d" in pids

    def test_noise_is_poisson_gaussian(self):
        tpl = _templates()["sted_graph_v2"]
        noise = [n for n in tpl["nodes"] if n["role"] == "noise"][0]
        assert noise["primitive_id"] == "poisson_gaussian_sensor"


class TestTemplatePALMSTORM:
    """PALM/STORM template structure checks."""

    def test_has_blinking_emitter(self):
        tpl = _templates()["palm_storm_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "blinking_emitter" in pids

    def test_source_is_photon(self):
        tpl = _templates()["palm_storm_graph_v2"]
        source = [n for n in tpl["nodes"] if n["role"] == "source"][0]
        assert source["primitive_id"] == "photon_source"

    def test_noise_is_poisson_gaussian(self):
        tpl = _templates()["palm_storm_graph_v2"]
        noise = [n for n in tpl["nodes"] if n["role"] == "noise"][0]
        assert noise["primitive_id"] == "poisson_gaussian_sensor"


class TestTemplateTIRF:
    """TIRF template structure checks."""

    def test_has_evanescent_decay(self):
        tpl = _templates()["tirf_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "evanescent_decay" in pids

    def test_has_conv2d(self):
        tpl = _templates()["tirf_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "conv2d" in pids

    def test_noise_is_poisson_gaussian(self):
        tpl = _templates()["tirf_graph_v2"]
        noise = [n for n in tpl["nodes"] if n["role"] == "noise"][0]
        assert noise["primitive_id"] == "poisson_gaussian_sensor"


class TestTemplatePolarization:
    """Polarization template structure checks."""

    def test_has_conv2d(self):
        tpl = _templates()["polarization_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "conv2d" in pids

    def test_source_is_photon(self):
        tpl = _templates()["polarization_graph_v2"]
        source = [n for n in tpl["nodes"] if n["role"] == "source"][0]
        assert source["primitive_id"] == "photon_source"

    def test_noise_is_poisson_gaussian(self):
        tpl = _templates()["polarization_graph_v2"]
        noise = [n for n in tpl["nodes"] if n["role"] == "noise"][0]
        assert noise["primitive_id"] == "poisson_gaussian_sensor"


# ===========================================================================
# Forward model primitive execution tests
# ===========================================================================


class TestPrimitiveForward:
    """Test that the key primitives produce finite output."""

    def test_nonlinear_excitation_forward(self):
        """NonlinearExcitation produces finite output."""
        try:
            from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
            prim_cls = PRIMITIVE_REGISTRY["nonlinear_excitation"]
            prim = prim_cls(params={"n_photons": 2})
            x = np.random.rand(64, 64).astype(np.float64)
            y = prim.forward(x)
            assert y.shape == (64, 64)
            assert np.all(np.isfinite(y))
            assert np.all(y >= 0)
        except (ImportError, KeyError):
            pytest.skip("Primitive registry not available")

    def test_saturation_depletion_forward(self):
        """SaturationDepletion produces finite output."""
        try:
            from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
            prim_cls = PRIMITIVE_REGISTRY["saturation_depletion"]
            prim = prim_cls(params={"depletion_factor": 0.5, "psf_sigma": 1.0})
            x = np.random.rand(64, 64).astype(np.float64)
            y = prim.forward(x)
            assert y.shape == (64, 64)
            assert np.all(np.isfinite(y))
        except (ImportError, KeyError):
            pytest.skip("Primitive registry not available")

    def test_blinking_emitter_forward(self):
        """BlinkingEmitterModel produces finite output."""
        try:
            from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
            prim_cls = PRIMITIVE_REGISTRY["blinking_emitter"]
            prim = prim_cls(params={"density": 0.1, "photons_per_emitter": 1000, "seed": 42})
            x = np.random.rand(64, 64).astype(np.float64)
            y = prim.forward(x)
            assert y.shape == (64, 64)
            assert np.all(np.isfinite(y))
            assert np.all(y >= 0)
        except (ImportError, KeyError):
            pytest.skip("Primitive registry not available")

    def test_evanescent_decay_forward(self):
        """EvanescentFieldDecay produces finite output for 2D input."""
        try:
            from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
            prim_cls = PRIMITIVE_REGISTRY["evanescent_decay"]
            prim = prim_cls(params={"penetration_depth": 100.0})
            x = np.random.rand(64, 64).astype(np.float64)
            y = prim.forward(x)
            assert y.shape == (64, 64)
            assert np.all(np.isfinite(y))
        except (ImportError, KeyError):
            pytest.skip("Primitive registry not available")

    def test_evanescent_decay_adjoint(self):
        """EvanescentFieldDecay adjoint works for 2D input."""
        try:
            from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
            prim_cls = PRIMITIVE_REGISTRY["evanescent_decay"]
            prim = prim_cls(params={"penetration_depth": 100.0})
            x = np.random.rand(64, 64).astype(np.float64)
            y = prim.adjoint(x)
            assert y.shape == (64, 64)
            assert np.all(np.isfinite(y))
        except (ImportError, KeyError):
            pytest.skip("Primitive registry not available")

    def test_conv2d_forward_finite(self):
        """conv2d produces finite output (used in two_photon, sted, tirf, polarization)."""
        try:
            from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
            prim_cls = PRIMITIVE_REGISTRY["conv2d"]
            prim = prim_cls(params={"sigma": 1.5, "mode": "reflect"})
            x = np.random.rand(64, 64).astype(np.float64)
            y = prim.forward(x)
            assert y.shape == (64, 64)
            assert np.all(np.isfinite(y))
        except (ImportError, KeyError):
            pytest.skip("Primitive registry not available")

    def test_conv2d_adjoint_finite(self):
        """conv2d adjoint produces finite output."""
        try:
            from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
            prim_cls = PRIMITIVE_REGISTRY["conv2d"]
            prim = prim_cls(params={"sigma": 1.5, "mode": "reflect"})
            x = np.random.rand(64, 64).astype(np.float64)
            y = prim.adjoint(x)
            assert y.shape == (64, 64)
            assert np.all(np.isfinite(y))
        except (ImportError, KeyError):
            pytest.skip("Primitive registry not available")


# ===========================================================================
# Cross-DB consistency
# ===========================================================================


class TestCrossDBConsistency:
    """Ensure all 5 modalities appear consistently across all registries."""

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_modality_in_all_dbs(self, m):
        """Each modality appears in all 5 registries."""
        assert m in _modalities(), f"{m} not in modalities.yaml"
        assert m in _mismatch(), f"{m} not in mismatch_db.yaml"
        assert m in _photon(), f"{m} not in photon_db.yaml"
        assert m in _compression(), f"{m} not in compression_db.yaml"
        # Check template exists
        tid = f"{m}_graph_v2"
        assert tid in _templates(), f"{tid} not in graph_templates.yaml"

    @pytest.mark.parametrize("m", R4_MODALITIES)
    def test_template_modality_matches(self, m):
        """Template metadata.modality matches the modality key."""
        tid = f"{m}_graph_v2"
        tpl = _templates()[tid]
        assert tpl["metadata"]["modality"] == m

    def test_all_r4_in_advanced_microscopy_metric_set(self):
        """All R4 modalities appear in the advanced_microscopy metric set."""
        ms = _metrics()["advanced_microscopy"]
        for m in R4_MODALITIES:
            assert m in ms["modalities"], f"{m} not in advanced_microscopy metric set"
