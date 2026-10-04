"""The imaging catalog and the design checker (needs the full pwm_core at PWM_CORE_SRC)."""
import pytest

from pwm_mode.imaging import catalog, core, design
from pwm_mode.library import Library, LibraryError

pytest.importorskip("yaml")
if not core.PWM_CORE_SRC.exists():
    pytest.skip("pwm_core not available", allow_module_level=True)

BASE = {"carrier": "photon", "geometry": {}, "object": {}, "noise": ["poisson"],
        "target": {"metric": "PSNR_dB", "min_snr": 20, "min_compression_ratio": 0.1},
        "system_elements": {"source": {"photons_per_pixel": 2000},
                            "detector": {"quantum_efficiency": 0.8, "read_noise_e": 2.0},
                            "calibration": [{"node": "w", "parameter": "disp_step", "tolerance": 0.2}]}}
SPECTRAL = {"nodes": [{"node_id": "m", "primitive_id": "coded_mask", "params": {"seed": 1, "H": 32, "W": 32}},
                      {"node_id": "w", "primitive_id": "spectral_dispersion", "params": {"disp_step": 1.0}},
                      {"node_id": "s", "primitive_id": "frame_integration", "params": {"axis": -1, "T": 4}}],
            "edges": [{"source": "m", "target": "w"}, {"source": "w", "target": "s"}],
            "metadata": {"x_shape": [32, 32, 4]}}


def test_catalog_covers_registry():
    cat = catalog.load()
    assert cat["summary"]["registry_modalities"] == 170
    assert cat["summary"]["runs"] >= 165
    ids = {e["id"] for e in cat["modalities"]}
    assert {"cassi", "mri", "ct", "ultrasound"} <= ids
    for e in cat["modalities"]:
        assert set(design.SPEC_FIELDS) <= set(e["spec_md"])
        assert e["level"] in ("registered", "runs", "canonical", "real_data")


def test_catalog_search_and_levels():
    rows = catalog.search("spectral", min_level="runs")
    assert rows and all(r["level"] != "registered" for r in rows)
    assert catalog.get("cassi")["level"] == "real_data"
    with pytest.raises(KeyError):
        catalog.get("no-such-modality")


def test_design_check_passes_and_recognises_known_chain():
    r = design.check_design(dict(BASE, modality="spectral_test", forward_model={"graph": SPECTRAL}))
    assert r["passed_structural"]
    assert r["checks"]["C5_adjoint"]["pass"]
    assert r["checks"]["C2_chain"]["matches_validated"] == ["cassi"]  # the same chain as CASSI, Detect implicit
    assert r["gates"]["G2_carrier_budget"]["pass"] is True
    assert r["gates"]["G3_mismatch"]["sensitivity"][0]["relative_output_change"] > 0
    assert any("gain" in b for b in r["gates"]["G3_mismatch"]["not_blindly_calibratable"])
    assert r["adequacy"]["budget_feasibility"] == "unassessed"  # never invented


def test_design_check_reports_missing_inputs_as_unassessed():
    spec = dict(BASE, modality="x", forward_model={"graph": SPECTRAL}, system_elements={}, target={})
    r = design.check_design(spec)
    assert r["gates"]["G2_carrier_budget"]["status"].startswith("unassessed")
    assert r["gates"]["G1_recoverability"]["pass"] is None


def test_design_check_rejects_unknown_primitive_and_flags_dead_parameter():
    bad = {"nodes": [{"node_id": "a", "primitive_id": "no_such_prim", "params": {}}], "edges": [], "metadata": {}}
    assert not design.check_design(dict(BASE, modality="bad", forward_model={"graph": bad}))["passed_structural"]
    se = dict(BASE["system_elements"], calibration=[{"node": "m", "parameter": "not_a_param", "tolerance": 1.0}])
    r = design.check_design(dict(BASE, modality="x", forward_model={"graph": SPECTRAL}, system_elements=se))
    assert "no effect" in r["gates"]["G3_mismatch"]["sensitivity"][0]["status"]


def test_alias_repairs_unregistered_primitive_names():
    g = {"nodes": [{"node_id": "p", "primitive_id": "propagation_2d", "params": {}}], "edges": [],
         "metadata": {"x_shape": [32, 32]}}
    r = design.check_design(dict(BASE, modality="x", forward_model={"graph": g}))
    assert r["checks"]["C1_dag"]["pass"] and any("propagation_2d" in n for n in r["notes"])


def test_submitted_design_needs_an_independent_validator(tmp_path):
    lib = Library(tmp_path / "lib")
    lib.seed([{"id": "imaging-root", "layer": "L1", "field": "imaging", "title": "root", "summary": ""}])
    spec = dict(BASE, modality="spectral_test", forward_model={"graph": SPECTRAL})
    rep = design.check_design(spec)
    sub = design.submit_design(spec, rep, "user:alice", lib)
    assert sub["state"] == "pending"
    with pytest.raises(LibraryError):
        lib.accept(sub["id"], {"passed": True, "validator": "user:alice"})
    st = lib.accept(sub["id"], {"passed": True, "validator": "validator-imaging-1", "metrics": {"c5": True}})
    assert any(r.startswith("img-design-spectral-test@") for r in st["artifacts"])
