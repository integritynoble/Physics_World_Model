"""Tests for R7 (Radar/Sonar), R8 (Electron spectroscopy/diffraction), R9 (Particle imaging).

Covers template compilation, YAML registry presence, forward model execution,
output shape, and DB consistency for all 9 new modalities:
  R7: sar, sonar
  R8: electron_diffraction, ebsd, eels, electron_holography
  R9: neutron_tomo, proton_radiography, muon_tomo
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


# The 9 new modality keys
R7_MODALITIES = ["sar", "sonar"]
R8_MODALITIES = ["electron_diffraction", "ebsd", "eels", "electron_holography"]
R9_MODALITIES = ["neutron_tomo", "proton_radiography", "muon_tomo"]
ALL_NEW = R7_MODALITIES + R8_MODALITIES + R9_MODALITIES

# Template IDs
TEMPLATE_IDS = [f"{m}_graph_v2" for m in ALL_NEW]


# ===========================================================================
# YAML structure tests
# ===========================================================================


class TestGraphTemplatesPresent:
    """All 9 new templates exist in graph_templates.yaml."""

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


class TestModalitiesPresent:
    """All 9 modalities in modalities.yaml."""

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

    @pytest.mark.parametrize("m", R7_MODALITIES)
    def test_radar_sonar_category(self, m):
        mod = _modalities()[m]
        assert mod["category"] == "remote_sensing"

    @pytest.mark.parametrize("m", R8_MODALITIES)
    def test_electron_category(self, m):
        mod = _modalities()[m]
        assert mod["category"] == "electron_microscopy"

    @pytest.mark.parametrize("m", R9_MODALITIES)
    def test_particle_category(self, m):
        mod = _modalities()[m]
        assert mod["category"] == "particle_imaging"

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
    """All 9 modalities in mismatch_db.yaml."""

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
    """All 9 modalities in photon_db.yaml."""

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
    """All 9 modalities in compression_db.yaml."""

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

    def test_radar_sonar_metric_set(self):
        ms = _metrics()
        assert "radar_sonar" in ms
        for m in R7_MODALITIES:
            assert m in ms["radar_sonar"]["modalities"]

    def test_electron_spectroscopy_metric_set(self):
        ms = _metrics()
        assert "electron_spectroscopy" in ms
        for m in R8_MODALITIES:
            assert m in ms["electron_spectroscopy"]["modalities"]

    def test_particle_imaging_metric_set(self):
        ms = _metrics()
        assert "particle_imaging" in ms
        for m in R9_MODALITIES:
            assert m in ms["particle_imaging"]["modalities"]


# ===========================================================================
# Template structure consistency tests
# ===========================================================================


class TestTemplateRadarSonar:
    """R7 Radar/Sonar template structure checks."""

    def test_sar_has_backprojection(self):
        tpl = _templates()["sar_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "sar_backprojection" in pids

    def test_sar_source_is_generic(self):
        tpl = _templates()["sar_graph_v2"]
        source = [n for n in tpl["nodes"] if n["role"] == "source"][0]
        assert source["primitive_id"] == "generic_source"

    def test_sar_noise_is_complex_gaussian(self):
        tpl = _templates()["sar_graph_v2"]
        noise = [n for n in tpl["nodes"] if n["role"] == "noise"][0]
        assert noise["primitive_id"] == "complex_gaussian_sensor"

    def test_sonar_has_beamform(self):
        tpl = _templates()["sonar_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "beamform_delay" in pids

    def test_sonar_source_is_acoustic(self):
        tpl = _templates()["sonar_graph_v2"]
        source = [n for n in tpl["nodes"] if n["role"] == "source"][0]
        assert source["primitive_id"] == "acoustic_source"


class TestTemplateElectronSpectroscopy:
    """R8 Electron spectroscopy/diffraction template structure checks."""

    def test_electron_diffraction_has_diffraction_camera(self):
        tpl = _templates()["electron_diffraction_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "diffraction_camera" in pids

    def test_ebsd_has_reciprocal_space_geometry(self):
        tpl = _templates()["ebsd_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "reciprocal_space_geometry" in pids

    def test_eels_has_energy_resolving_detector(self):
        tpl = _templates()["eels_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "energy_resolving_detector" in pids

    def test_electron_holography_has_fresnel_prop(self):
        tpl = _templates()["electron_holography_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "fresnel_prop" in pids

    def test_electron_holography_sensor_is_photon(self):
        tpl = _templates()["electron_holography_graph_v2"]
        sensors = [n for n in tpl["nodes"] if n["role"] == "sensor"]
        assert sensors[0]["primitive_id"] == "photon_sensor"


class TestTemplateParticleImaging:
    """R9 Particle imaging template structure checks."""

    def test_neutron_tomo_has_particle_attenuation(self):
        tpl = _templates()["neutron_tomo_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "particle_attenuation" in pids

    def test_proton_radiography_has_multiple_scattering(self):
        tpl = _templates()["proton_radiography_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "multiple_scattering" in pids

    def test_proton_radiography_has_particle_attenuation(self):
        tpl = _templates()["proton_radiography_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "particle_attenuation" in pids

    def test_muon_tomo_has_multiple_scattering(self):
        tpl = _templates()["muon_tomo_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "multiple_scattering" in pids

    def test_muon_tomo_has_track_detector(self):
        tpl = _templates()["muon_tomo_graph_v2"]
        sensors = [n for n in tpl["nodes"] if n["role"] == "sensor"]
        assert sensors[0]["primitive_id"] == "track_detector_sensor"


# ===========================================================================
# Forward model execution tests
# ===========================================================================


class TestForwardExecution:
    """Test that forward model primitives produce finite output for each modality."""

    def _get_primitives_chain(self, template_id):
        """Build a chain of primitive forward calls from a template."""
        from pwm_core.graph.primitives import get_primitive

        tpl = _templates()[template_id]
        nodes = {n["node_id"]: n for n in tpl["nodes"]}
        edges = tpl["edges"]

        # Build execution order from edges
        order = []
        visited = set()
        # Find source node (no incoming edges)
        targets = {e["target"] for e in edges}
        sources = {e["source"] for e in edges}
        start_nodes = sources - targets
        if not start_nodes:
            start_nodes = {tpl["nodes"][0]["node_id"]}

        # BFS to get topological order
        queue = list(start_nodes)
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            order.append(nid)
            for e in edges:
                if e["source"] == nid:
                    queue.append(e["target"])

        primitives = []
        for nid in order:
            node = nodes[nid]
            prim = get_primitive(node["primitive_id"], node.get("params", {}))
            primitives.append(prim)
        return primitives

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_forward_produces_finite(self, m):
        """Run forward through all primitives and check output is finite."""
        from pwm_core.graph.primitives import get_primitive

        tid = f"{m}_graph_v2"
        tpl = _templates()[tid]
        x_shape = tuple(tpl["metadata"]["x_shape"])
        x = np.random.RandomState(42).rand(*x_shape).astype(np.float64)
        x = np.clip(x, 0.01, 1.0)  # Ensure positive for Beer-Lambert

        primitives = self._get_primitives_chain(tid)
        signal = x
        for prim in primitives:
            signal = prim.forward(signal)
            if isinstance(signal, np.ndarray):
                signal = np.asarray(signal, dtype=np.float64) if not np.iscomplexobj(signal) else signal

        result = np.asarray(signal)
        assert np.all(np.isfinite(result)), f"{m}: forward produced non-finite output"
        assert result.size > 0, f"{m}: forward produced empty output"

    @pytest.mark.parametrize("m", ALL_NEW)
    def test_output_shape_nonzero(self, m):
        """Check that the output has a non-trivial shape."""
        from pwm_core.graph.primitives import get_primitive

        tid = f"{m}_graph_v2"
        tpl = _templates()[tid]
        x_shape = tuple(tpl["metadata"]["x_shape"])
        x = np.random.RandomState(42).rand(*x_shape).astype(np.float64)
        x = np.clip(x, 0.01, 1.0)

        primitives = self._get_primitives_chain(tid)
        signal = x
        for prim in primitives:
            signal = prim.forward(signal)

        result = np.asarray(signal)
        assert result.ndim >= 1, f"{m}: output should have at least 1 dimension"
        assert all(s > 0 for s in result.shape), f"{m}: output shape has zero dim: {result.shape}"
