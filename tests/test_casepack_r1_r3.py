"""Tests for R1 (X-ray variants), R2 (Ultrasound modes), R3 (MRI applications).

Covers template compilation, YAML registry presence, and DB consistency
for all 10 new modalities: fluoroscopy, mammography, dexa, cbct, angiography,
doppler_ultrasound, elastography, fmri, mrs, diffusion_mri.
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


# The 10 new modality keys
R1_MODALITIES = ["fluoroscopy", "mammography", "dexa", "cbct", "angiography"]
R2_MODALITIES = ["doppler_ultrasound", "elastography"]
R3_MODALITIES = ["fmri", "mrs", "diffusion_mri"]
ALL_NEW = R1_MODALITIES + R2_MODALITIES + R3_MODALITIES

# Template IDs
TEMPLATE_IDS = [f"{m}_graph_v2" for m in ALL_NEW]


# ===========================================================================
# YAML structure tests
# ===========================================================================


class TestGraphTemplatesPresent:
    """All 10 new templates exist in graph_templates.yaml."""

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
        # sensor may be named sensor
        assert "sensor" in roles, f"No sensor role in {tid}"

    @pytest.mark.parametrize("tid", TEMPLATE_IDS)
    def test_edge_targets_are_valid_node_ids(self, tid):
        tpl = _templates()[tid]
        node_ids = {n["node_id"] for n in tpl["nodes"]}
        for edge in tpl["edges"]:
            assert edge["source"] in node_ids, f"Edge source {edge['source']} not a node"
            assert edge["target"] in node_ids, f"Edge target {edge['target']} not a node"


class TestModalitiesPresent:
    """All 10 modalities in modalities.yaml."""

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_modality_exists(self, m):
        mods = _modalities()
        assert m in mods, f"Modality {m} missing from modalities.yaml"

    @pytest.mark.parametrize("m", ALL_NEW)
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

    @pytest.mark.parametrize("m", R1_MODALITIES)
    def test_xray_category(self, m):
        mod = _modalities()[m]
        assert mod["category"] == "medical"

    @pytest.mark.parametrize("m", R2_MODALITIES)
    def test_ultrasound_category(self, m):
        mod = _modalities()[m]
        assert mod["category"] == "medical_ultrasound"

    @pytest.mark.parametrize("m", R3_MODALITIES)
    def test_mri_category(self, m):
        mod = _modalities()[m]
        assert mod["category"] == "medical"

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_template_id_matches(self, m):
        mod = _modalities()[m]
        expected = f"{m}_graph_v2"
        assert mod["default_template_id"] == expected

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_elements_have_source_and_detector(self, m):
        elements = _modalities()[m]["elements"]
        types = [e["element_type"] for e in elements]
        assert "source" in types, f"{m}: no source element"
        assert "detector" in types, f"{m}: no detector element"


class TestMismatchDBPresent:
    """All 10 modalities in mismatch_db.yaml."""

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_modality_in_mismatch(self, m):
        db = _mismatch()
        assert m in db, f"Modality {m} missing from mismatch_db.yaml"

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_mismatch_has_parameters(self, m):
        entry = _mismatch()[m]
        assert "parameters" in entry
        assert len(entry["parameters"]) >= 2, f"{m}: need at least 2 mismatch params"

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_severity_weights_sum(self, m):
        entry = _mismatch()[m]
        weights = entry["severity_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.05, f"{m}: severity weights sum to {total}"

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_correction_method_exists(self, m):
        entry = _mismatch()[m]
        assert "correction_method" in entry
        assert entry["correction_method"] in [
            "grid_search",
            "gradient_descent",
            "UPWMI_beam_search",
        ]

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_param_fields(self, m):
        entry = _mismatch()[m]
        for pname, pval in entry["parameters"].items():
            for field in ["range", "typical_error", "unit", "param_type", "description"]:
                assert field in pval, f"{m}.{pname} missing {field}"


class TestPhotonDBPresent:
    """All 10 modalities in photon_db.yaml."""

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_modality_in_photon(self, m):
        db = _photon()
        assert m in db, f"Modality {m} missing from photon_db.yaml"

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_photon_levels(self, m):
        entry = _photon()[m]
        assert "photon_levels" in entry
        for level in ["bright", "standard", "low_light"]:
            assert level in entry["photon_levels"], f"{m}: missing photon level {level}"

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_model_id_exists(self, m):
        entry = _photon()[m]
        assert "model_id" in entry
        assert isinstance(entry["model_id"], str)

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_noise_model_exists(self, m):
        entry = _photon()[m]
        assert "noise_model" in entry
        assert entry["noise_model"] in [
            "poisson",
            "poisson_gaussian",
            "gaussian",
        ]


class TestCompressionDBPresent:
    """All 10 modalities in compression_db.yaml."""

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_modality_in_compression(self, m):
        db = _compression()
        assert m in db, f"Modality {m} missing from compression_db.yaml"

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_has_entries(self, m):
        entry = _compression()[m]
        assert "entries" in entry
        assert len(entry["entries"]) >= 2, f"{m}: need at least 2 calibration entries"

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_signal_prior_class(self, m):
        entry = _compression()[m]
        assert "signal_prior_class" in entry

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_entry_fields(self, m):
        entries = _compression()[m]["entries"]
        for i, e in enumerate(entries):
            for field in ["cr", "noise", "solver", "recoverability", "expected_psnr_db", "provenance"]:
                assert field in e, f"{m} entry[{i}] missing {field}"

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_provenance_fields(self, m):
        entries = _compression()[m]["entries"]
        for i, e in enumerate(entries):
            prov = e["provenance"]
            for field in ["dataset_id", "seed_set", "operator_version", "solver_version", "date_generated"]:
                assert field in prov, f"{m} entry[{i}] provenance missing {field}"


class TestMetricsDBPresent:
    """New modalities are covered in metrics_db.yaml metric sets."""

    def test_xray_variants_metric_set(self):
        ms = _metrics()
        assert "xray_variants" in ms
        for m in R1_MODALITIES:
            assert m in ms["xray_variants"]["modalities"]

    def test_ultrasound_advanced_metric_set(self):
        ms = _metrics()
        assert "ultrasound_advanced" in ms
        for m in R2_MODALITIES:
            assert m in ms["ultrasound_advanced"]["modalities"]

    def test_mri_advanced_metric_set(self):
        ms = _metrics()
        assert "mri_advanced" in ms
        for m in R3_MODALITIES:
            assert m in ms["mri_advanced"]["modalities"]


# ===========================================================================
# Template structure consistency tests
# ===========================================================================


class TestTemplateXRayVariants:
    """R1 X-ray variant template structure checks."""

    def test_fluoroscopy_has_temporal_integrator(self):
        tpl = _templates()["fluoroscopy_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "fluoro_temporal_integrator" in pids

    def test_fluoroscopy_has_beer_lambert(self):
        tpl = _templates()["fluoroscopy_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "beer_lambert" in pids

    def test_mammography_source_is_xray(self):
        tpl = _templates()["mammography_graph_v2"]
        source = [n for n in tpl["nodes"] if n["role"] == "source"][0]
        assert source["primitive_id"] == "xray_source"

    def test_dexa_has_dual_energy(self):
        tpl = _templates()["dexa_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "dual_energy_beer_lambert" in pids

    def test_cbct_has_radon(self):
        tpl = _templates()["cbct_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "ct_radon" in pids

    def test_cbct_noise_is_poisson_only(self):
        tpl = _templates()["cbct_graph_v2"]
        noise = [n for n in tpl["nodes"] if n["role"] == "noise"][0]
        assert noise["primitive_id"] == "poisson_only_sensor"

    def test_angiography_has_temporal_integrator(self):
        tpl = _templates()["angiography_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "fluoro_temporal_integrator" in pids


class TestTemplateUltrasoundModes:
    """R2 Ultrasound mode template structure checks."""

    def test_doppler_has_acoustic_source(self):
        tpl = _templates()["doppler_ultrasound_graph_v2"]
        source = [n for n in tpl["nodes"] if n["role"] == "source"][0]
        assert source["primitive_id"] == "acoustic_source"

    def test_doppler_has_doppler_estimator(self):
        tpl = _templates()["doppler_ultrasound_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "doppler_estimator" in pids

    def test_elastography_has_elastic_wave_model(self):
        tpl = _templates()["elastography_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "elastic_wave_model" in pids

    def test_elastography_noise_is_gaussian(self):
        tpl = _templates()["elastography_graph_v2"]
        noise = [n for n in tpl["nodes"] if n["role"] == "noise"][0]
        assert noise["primitive_id"] == "gaussian_sensor_noise"


class TestTemplateMRIApplications:
    """R3 MRI application template structure checks."""

    def test_fmri_has_spin_source(self):
        tpl = _templates()["fmri_graph_v2"]
        source = [n for n in tpl["nodes"] if n["role"] == "source"][0]
        assert source["primitive_id"] == "spin_source"

    def test_fmri_has_sequence_block(self):
        tpl = _templates()["fmri_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "sequence_block" in pids

    def test_fmri_has_coil_sensor(self):
        tpl = _templates()["fmri_graph_v2"]
        sensors = [n for n in tpl["nodes"] if n["role"] == "sensor"]
        assert sensors[0]["primitive_id"] == "coil_sensor"

    def test_mrs_has_sequence_block(self):
        tpl = _templates()["mrs_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "sequence_block" in pids

    def test_mrs_noise_is_complex_gaussian(self):
        tpl = _templates()["mrs_graph_v2"]
        noise = [n for n in tpl["nodes"] if n["role"] == "noise"][0]
        assert noise["primitive_id"] == "complex_gaussian_sensor"

    def test_diffusion_mri_has_kspace(self):
        tpl = _templates()["diffusion_mri_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "mri_kspace" in pids

    def test_diffusion_mri_no_sequence_block(self):
        """Diffusion MRI template does not include sequence_block (simpler chain)."""
        tpl = _templates()["diffusion_mri_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "sequence_block" not in pids

    def test_diffusion_mri_noise_is_complex_gaussian(self):
        tpl = _templates()["diffusion_mri_graph_v2"]
        noise = [n for n in tpl["nodes"] if n["role"] == "noise"][0]
        assert noise["primitive_id"] == "complex_gaussian_sensor"
