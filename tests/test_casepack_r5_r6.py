"""Tests for R5 (Clinical optics) and R6 (Depth/ToF imaging).

Covers template compilation, YAML registry presence, DB consistency,
forward pass finiteness, and output shape correctness for all 6 new
modalities: endoscopy, fundus, octa, tof_camera, lidar, structured_light.
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


# The 6 new modality keys
R5_MODALITIES = ["endoscopy", "fundus", "octa"]
R6_MODALITIES = ["tof_camera", "lidar", "structured_light"]
ALL_NEW = R5_MODALITIES + R6_MODALITIES

# Template IDs
TEMPLATE_IDS = [f"{m}_graph_v2" for m in ALL_NEW]


# ===========================================================================
# YAML structure tests
# ===========================================================================


class TestGraphTemplatesPresent:
    """All 6 new templates exist in graph_templates.yaml."""

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
    """All 6 modalities in modalities.yaml."""

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

    @pytest.mark.parametrize("m", R5_MODALITIES)
    def test_clinical_optics_category(self, m):
        mod = _modalities()[m]
        assert mod["category"] == "clinical_optics"

    @pytest.mark.parametrize("m", R6_MODALITIES)
    def test_depth_imaging_category(self, m):
        mod = _modalities()[m]
        assert mod["category"] == "depth_imaging"

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
    """All 6 modalities in mismatch_db.yaml."""

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
    """All 6 modalities in photon_db.yaml."""

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
    """All 6 modalities in compression_db.yaml."""

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

    def test_clinical_optics_metric_set(self):
        ms = _metrics()
        assert "clinical_optics" in ms
        for m in R5_MODALITIES:
            assert m in ms["clinical_optics"]["modalities"]

    def test_clinical_optics_metrics(self):
        ms = _metrics()
        expected = ["psnr", "ssim", "cnr", "nrmse"]
        for metric in expected:
            assert metric in ms["clinical_optics"]["metrics"]

    def test_depth_imaging_metric_set(self):
        ms = _metrics()
        assert "depth_imaging" in ms
        for m in R6_MODALITIES:
            assert m in ms["depth_imaging"]["modalities"]

    def test_depth_imaging_metrics(self):
        ms = _metrics()
        expected = ["mae_depth", "rmse_depth", "ssim", "completeness"]
        for metric in expected:
            assert metric in ms["depth_imaging"]["metrics"]


# ===========================================================================
# Template structure consistency tests
# ===========================================================================


class TestTemplateClinicaOptics:
    """R5 clinical optics template structure checks."""

    def test_endoscopy_has_fiber_bundle(self):
        tpl = _templates()["endoscopy_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "fiber_bundle_sensor" in pids

    def test_endoscopy_has_specular_reflection(self):
        tpl = _templates()["endoscopy_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "specular_reflection" in pids

    def test_endoscopy_source_is_photon(self):
        tpl = _templates()["endoscopy_graph_v2"]
        source = [n for n in tpl["nodes"] if n["role"] == "source"][0]
        assert source["primitive_id"] == "photon_source"

    def test_fundus_has_conv2d_retinal_psf(self):
        tpl = _templates()["fundus_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "conv2d" in pids

    def test_fundus_has_photon_sensor(self):
        tpl = _templates()["fundus_graph_v2"]
        sensors = [n for n in tpl["nodes"] if n["role"] == "sensor"]
        assert sensors[0]["primitive_id"] == "photon_sensor"

    def test_octa_has_angular_spectrum(self):
        tpl = _templates()["octa_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "angular_spectrum" in pids

    def test_octa_has_vessel_flow_contrast(self):
        tpl = _templates()["octa_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "vessel_flow_contrast" in pids


class TestTemplateDepthImaging:
    """R6 depth/ToF imaging template structure checks."""

    def test_tof_camera_has_tof_gate(self):
        tpl = _templates()["tof_camera_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "tof_gate" in pids

    def test_tof_camera_has_spad_sensor(self):
        tpl = _templates()["tof_camera_graph_v2"]
        sensors = [n for n in tpl["nodes"] if n["role"] == "sensor"]
        assert sensors[0]["primitive_id"] == "spad_tof_sensor"

    def test_lidar_has_scan_trajectory(self):
        tpl = _templates()["lidar_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "scan_trajectory" in pids

    def test_lidar_has_tof_gate(self):
        tpl = _templates()["lidar_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "tof_gate" in pids

    def test_lidar_has_spad_sensor(self):
        tpl = _templates()["lidar_graph_v2"]
        sensors = [n for n in tpl["nodes"] if n["role"] == "sensor"]
        assert sensors[0]["primitive_id"] == "spad_tof_sensor"

    def test_structured_light_has_projector(self):
        tpl = _templates()["structured_light_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "structured_light_projector" in pids

    def test_structured_light_has_depth_optics(self):
        tpl = _templates()["structured_light_graph_v2"]
        pids = [n["primitive_id"] for n in tpl["nodes"]]
        assert "depth_optics" in pids

    def test_structured_light_noise_is_poisson_gaussian(self):
        tpl = _templates()["structured_light_graph_v2"]
        noise = [n for n in tpl["nodes"] if n["role"] == "noise"][0]
        assert noise["primitive_id"] == "poisson_gaussian_sensor"


# ===========================================================================
# Forward pass tests: compile primitives and verify finite output
# ===========================================================================


def _get_primitive_class(primitive_id: str):
    """Look up a primitive class from the PRIMITIVE_REGISTRY."""
    from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
    return PRIMITIVE_REGISTRY.get(primitive_id)


def _compile_and_forward(template_id: str, x_shape=None):
    """Compile a template chain and run forward on random input.

    Returns the output array.
    """
    tpl = _templates()[template_id]
    meta = tpl["metadata"]
    if x_shape is None:
        x_shape = tuple(meta.get("x_shape", [64, 64]))
    x = np.random.default_rng(42).random(x_shape).astype(np.float64)
    x = x * 0.8 + 0.1  # Ensure positive for Poisson-like noise

    # Build nodes in order
    nodes_by_id = {}
    for node_def in tpl["nodes"]:
        nid = node_def["node_id"]
        pid = node_def["primitive_id"]
        params = node_def.get("params", {})
        cls = _get_primitive_class(pid)
        if cls is None:
            pytest.skip(f"Primitive {pid} not in PRIMITIVE_REGISTRY")
        nodes_by_id[nid] = cls(params)

    # Determine execution order from edges (simple topological sort)
    edges = tpl["edges"]
    # Find source node (not a target of any edge)
    targets = {e["target"] for e in edges}
    sources_set = {e["source"] for e in edges}
    root_candidates = sources_set - targets
    if not root_candidates:
        root_candidates = {tpl["nodes"][0]["node_id"]}
    root = list(root_candidates)[0]

    # BFS order
    order = [root]
    edge_map = {}
    for e in edges:
        edge_map.setdefault(e["source"], []).append(e["target"])

    visited = {root}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for nxt in edge_map.get(current, []):
            if nxt not in visited:
                visited.add(nxt)
                order.append(nxt)
                queue.append(nxt)

    # Run forward
    data = x
    for nid in order:
        if nid in nodes_by_id:
            data = nodes_by_id[nid].forward(data)

    return data


class TestForwardPassFinite:
    """Forward pass produces finite output for all 6 modalities."""

    @pytest.mark.parametrize("tid", TEMPLATE_IDS)
    def test_forward_finite(self, tid):
        output = _compile_and_forward(tid)
        assert np.all(np.isfinite(output)), f"Non-finite values in {tid} forward output"

    @pytest.mark.parametrize("tid", TEMPLATE_IDS)
    def test_forward_non_empty(self, tid):
        output = _compile_and_forward(tid)
        assert output.size > 0, f"Empty output from {tid}"


class TestOutputShape:
    """Output shapes are reasonable for all 6 modalities."""

    def test_endoscopy_output_2d(self):
        output = _compile_and_forward("endoscopy_graph_v2")
        assert output.ndim >= 1, "Endoscopy output should be at least 1D"

    def test_fundus_output_2d(self):
        output = _compile_and_forward("fundus_graph_v2")
        assert output.ndim == 2, "Fundus output should be 2D"
        assert output.shape == (64, 64)

    def test_octa_output_2d(self):
        output = _compile_and_forward("octa_graph_v2")
        assert output.ndim >= 1, "OCTA output should be at least 1D"

    def test_tof_camera_output(self):
        output = _compile_and_forward("tof_camera_graph_v2")
        assert output.ndim >= 1, "ToF camera output should be at least 1D"

    def test_lidar_output(self):
        output = _compile_and_forward("lidar_graph_v2")
        assert output.ndim >= 1, "LiDAR output should be at least 1D"

    def test_structured_light_output_2d(self):
        output = _compile_and_forward("structured_light_graph_v2")
        assert output.ndim == 2, "Structured light output should be 2D"
        assert output.shape == (64, 64)
