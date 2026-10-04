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
    assert cat["summary"]["runs"] == 170  # IMG-01: was 165 with the five sampled-Fourier templates failing
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


FIVE = ["eht_imaging", "insar", "radio_astronomy", "radio_interferometry", "xray_crystallography"]


@pytest.mark.parametrize("mid", FIVE)
def test_img01_sampled_fourier_templates_compile_run_and_keep_the_imaginary_part(mid):
    import numpy as np
    e = catalog.get(mid)
    assert e["level"] in ("runs", "canonical", "real_data")
    (tid, rec), = e["templates"].items()
    assert rec["runs"] and not rec.get("error") and rec["aliases_applied"] == []
    t = core.contrib("graph_templates.yaml")["templates"][tid]
    assert [n["primitive_id"] for n in t["nodes"]] == ["mri_kspace", "complex_gaussian_sensor"]
    op = core.compile_graph(*core.normalise_nodes(t["nodes"])[:1], t["edges"], t["metadata"])
    y = op.forward(np.random.default_rng(0).random(tuple(t["metadata"]["x_shape"])))
    assert np.iscomplexobj(y) and np.all(np.isfinite(y)) and np.std(y.imag) > 0
    # about a quarter of the k-space samples plus the fully sampled centre block: far fewer than all, many more than none
    clean = core.registry()["mri_kspace"](t["nodes"][0]["params"]).forward(np.random.default_rng(3).random(y.shape))
    frac = float(np.mean(clean != 0))
    assert 0.25 < frac < 0.45


def test_img01_fft_2d_is_not_aliased_to_a_propagation_operator():
    assert "fft_2d" not in core.PRIMITIVE_ALIASES
    assert set(core.PRIMITIVE_ALIASES) == {"propagation_2d", "radon_2d", "poisson_gaussian_sensor_noise"}


def test_img01_repair_only_touches_templates_that_still_have_subsample():
    templates = core.contrib("graph_templates.yaml")["templates"]
    already_fine = templates["mri_graph_v1"]
    assert core.repair_template(already_fine) is already_fine
    patched = core.repair_template({"nodes": [{"node_id": "kspace", "primitive_id": "mri_kspace", "params": {}}]})
    assert patched["nodes"][0]["primitive_id"] == "mri_kspace"


def test_img01_pwm_core_patch_applies_to_unpatched_templates_and_fixes_all_unregistered_names(tmp_path):
    import shutil
    import subprocess
    import yaml
    patch = core.Path(__file__).resolve().parents[1] / "results/IMG-01/pwm_core_templates.patch"
    src = core.PWM_CORE_SRC / "contrib" / "graph_templates.yaml"
    if "primitive_id: subsample" not in src.read_text():
        pytest.skip("pwm_core at PWM_CORE_SRC already carries the IMG-01 fix")
    if shutil.which("git") is None:
        pytest.skip("git not available")
    dst = tmp_path / "packages/pwm_core/contrib"
    dst.mkdir(parents=True)
    shutil.copy(src, dst / "graph_templates.yaml")
    subprocess.run(["git", "apply", "-p1", str(patch)], cwd=tmp_path, check=True)
    after = yaml.safe_load((dst / "graph_templates.yaml").read_text())["templates"]
    used = {n["primitive_id"] for t in after.values() for n in t["nodes"]}
    assert not used & {"fft_2d", "subsample", "propagation_2d", "radon_2d", "poisson_gaussian_sensor_noise"}
    assert used <= set(core.registry())


@pytest.mark.xfail(strict=True, reason="IMG-07 finding: pwm_core mri_kspace.adjoint is the unit-normalised inverse FFT "
                                       "(zero-filled reconstruction), so the dot-product test fails by exactly H*W; "
                                       "applies to the MRI template too. Remove this mark when pwm_core fixes it.")
def test_img01_repaired_operator_passes_the_adjoint_dot_test():
    import numpy as np
    op = core.registry()["mri_kspace"]({"H": 64, "W": 64, "sampling_rate": 0.25, "seed": 42})
    rng = np.random.default_rng(0)
    x = rng.random((64, 64))
    y = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    assert np.isclose(np.real(np.vdot(y, op.forward(x))), float(np.sum(x * op.adjoint(y))), rtol=1e-6)
