"""IMG-02: drafted source elements for the 54 modalities that had none."""
import pytest

from pwm_mode.imaging import catalog, core, sources

pytest.importorskip("yaml")
if not core.PWM_CORE_SRC.exists():
    pytest.skip("pwm_core not available", allow_module_level=True)


def test_every_source_less_modality_has_a_drafted_source():
    cat = catalog.load()
    drafted = {e["id"] for e in cat["modalities"] if e["source_validation"] == "drafted"}
    assert len(drafted) == 54 and drafted == set(sources.SOURCES)
    assert cat["summary"]["with_source_element"] == 116 and cat["summary"]["without_source"] == 0
    for e in cat["modalities"]:
        assert e["spec_md"]["system_elements"]["source"], e["id"]
        assert (catalog.SOURCE_DRAFTED_GAP in e["gaps"]) == (e["source_validation"] == "drafted")


def test_drafted_sources_are_honestly_labelled_and_well_formed():
    for mid in sources.SOURCES:
        assert sources.check(mid) == [], mid
        for e in catalog.get(mid)["spec_md"]["system_elements"]["source"]:
            assert e["validation"] == "drafted" and e["verified_against"] is None
            assert e["parameters"], (mid, e["name"])


def test_source_primitives_are_registered_source_nodes():
    reg = core.registry()
    for mid, els in sources.SOURCES.items():
        for e in els:
            assert e["primitive"] in reg, (mid, e["primitive"])
            assert getattr(reg[e["primitive"]], "_node_role", None) == "source", (mid, e["primitive"])


def test_registry_sources_are_untouched_and_wavelength_conflicts_are_flagged():
    cat = catalog.load()
    cassi = catalog.get("cassi")
    assert cassi["source_validation"] == "registry" and "validation" not in cassi["spec_md"]["system_elements"]["source"][0]
    flagged = {e["id"] for e in cat["modalities"] if any(g.startswith("geometry wavelength_range_nm") for g in e["gaps"])}
    assert {"cars", "libs", "shg", "srs"} <= flagged
    assert "xray_crystallography" not in flagged


def test_check_catches_bad_entries():
    bad = dict(sources.SOURCES["libs"][0], parameters={"power": 3.0})
    sources.SOURCES["_x"] = [bad]
    try:
        assert any("no unit suffix" in m for m in sources.check("_x"))
        sources.SOURCES["_x"] = [dict(bad, parameters={"power_mw": 99}, constraints=[sources._c("power_mw", None, 50, "x")])]
        assert any("outside its own constraint" in m for m in sources.check("_x"))
    finally:
        del sources.SOURCES["_x"]
