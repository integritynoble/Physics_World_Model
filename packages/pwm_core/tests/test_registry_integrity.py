"""Registry integrity tests — fail on any orphan key, missing field, or cross-ref error."""

import os
import sys
import pytest
import yaml

# Add package to path
PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "pwm_core")
CONTRIB_DIR = os.path.join(os.path.dirname(__file__), "..", "contrib")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_yaml(filename):
    path = os.path.join(CONTRIB_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not found")
    with open(path) as f:
        return yaml.safe_load(f)


class TestRegistryIntegrity:
    """All YAML registries must be consistent."""

    def test_modalities_yaml_loads(self):
        data = load_yaml("modalities.yaml")
        assert "version" in data
        assert "modalities" in data
        assert len(data["modalities"]) >= 18

    def test_mismatch_db_loads(self):
        data = load_yaml("mismatch_db.yaml")
        assert "version" in data
        assert "modalities" in data

    def test_photon_db_loads(self):
        data = load_yaml("photon_db.yaml")
        assert "version" in data
        assert "modalities" in data

    def test_compression_db_loads(self):
        data = load_yaml("compression_db.yaml")
        assert "version" in data
        assert "calibration_tables" in data

    def test_metrics_db_loads(self):
        data = load_yaml("metrics_db.yaml")
        assert "version" in data
        assert "metric_sets" in data

    def test_no_orphan_mismatch_keys(self):
        mod_keys = set(load_yaml("modalities.yaml")["modalities"].keys())
        mismatch_keys = set(load_yaml("mismatch_db.yaml")["modalities"].keys())
        orphans = mismatch_keys - mod_keys
        assert not orphans, f"Orphan mismatch keys: {orphans}"

    def test_no_orphan_photon_keys(self):
        mod_keys = set(load_yaml("modalities.yaml")["modalities"].keys())
        photon_keys = set(load_yaml("photon_db.yaml")["modalities"].keys())
        orphans = photon_keys - mod_keys
        assert not orphans, f"Orphan photon keys: {orphans}"

    def test_no_orphan_compression_keys(self):
        mod_keys = set(load_yaml("modalities.yaml")["modalities"].keys())
        comp_keys = set(load_yaml("compression_db.yaml")["calibration_tables"].keys())
        orphans = comp_keys - mod_keys
        assert not orphans, f"Orphan compression keys: {orphans}"

    def test_every_modality_in_mismatch(self):
        mod_keys = set(load_yaml("modalities.yaml")["modalities"].keys())
        mismatch_keys = set(load_yaml("mismatch_db.yaml")["modalities"].keys())
        missing = mod_keys - mismatch_keys
        assert not missing, f"Modalities missing from mismatch_db: {missing}"

    def test_every_modality_in_photon(self):
        mod_keys = set(load_yaml("modalities.yaml")["modalities"].keys())
        photon_keys = set(load_yaml("photon_db.yaml")["modalities"].keys())
        missing = mod_keys - photon_keys
        assert not missing, f"Modalities missing from photon_db: {missing}"

    def test_every_modality_in_compression(self):
        mod_keys = set(load_yaml("modalities.yaml")["modalities"].keys())
        comp_keys = set(load_yaml("compression_db.yaml")["calibration_tables"].keys())
        missing = mod_keys - comp_keys
        assert not missing, f"Modalities missing from compression_db: {missing}"

    def test_metrics_modalities_valid(self):
        mod_keys = set(load_yaml("modalities.yaml")["modalities"].keys())
        metrics = load_yaml("metrics_db.yaml")
        for ms_name, ms in metrics["metric_sets"].items():
            for m in ms["modalities"]:
                assert m in mod_keys, f"metrics_db[{ms_name}] references unknown modality: {m}"

    def test_every_modality_has_elements(self):
        modalities = load_yaml("modalities.yaml")["modalities"]
        for key, mod in modalities.items():
            assert "elements" in mod, f"{key} missing elements"
            assert len(mod["elements"]) >= 2, f"{key} has fewer than 2 elements"

    def test_every_modality_has_detector(self):
        modalities = load_yaml("modalities.yaml")["modalities"]
        for key, mod in modalities.items():
            elem_types = [e.get("element_type", "") for e in mod["elements"]]
            has_detector = "detector" in elem_types or "transducer" in elem_types
            assert has_detector, f"{key} has no detector/transducer element"

    def test_compression_entries_have_provenance(self):
        comp = load_yaml("compression_db.yaml")
        for modality, table in comp["calibration_tables"].items():
            for i, entry in enumerate(table["entries"]):
                assert "provenance" in entry, \
                    f"Missing provenance: compression_db[{modality}].entries[{i}]"

    def test_severity_weights_sum_approximately_one(self):
        mismatch = load_yaml("mismatch_db.yaml")
        for key, mod in mismatch["modalities"].items():
            total = sum(mod["severity_weights"].values())
            assert 0.9 <= total <= 1.1, \
                f"{key} severity weights sum to {total}, expected ~1.0"

    def test_modality_schema_validation(self):
        """Validate modalities.yaml through Pydantic schema."""
        try:
            from pwm_core.agents.registry_schemas import ModalitiesFileYaml
            data = load_yaml("modalities.yaml")
            ModalitiesFileYaml.model_validate(data)
        except ImportError:
            pytest.skip("registry_schemas not available")

    def test_compression_schema_validation(self):
        """Validate compression_db.yaml through Pydantic schema."""
        try:
            from pwm_core.agents.registry_schemas import CompressionDbFileYaml
            data = load_yaml("compression_db.yaml")
            CompressionDbFileYaml.model_validate(data)
        except ImportError:
            pytest.skip("registry_schemas not available")
