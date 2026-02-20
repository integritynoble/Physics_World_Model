"""Comprehensive tests for the scanner_registry module.

Tests cover the ScannerModelInfo / ScannerInstance Pydantic models and
the ScannerRegistry class (built-in lookup, case-insensitive search,
instance management, and dynamic model registration).
"""

from __future__ import annotations

import pytest

from pwm_core.clinical.common.scanner_registry import (
    ScannerRegistry,
    ScannerModelInfo,
    ScannerInstance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def registry() -> ScannerRegistry:
    """Return a freshly initialised ScannerRegistry."""
    return ScannerRegistry()


@pytest.fixture()
def sample_model_info() -> ScannerModelInfo:
    """A hand-crafted ScannerModelInfo for testing registration."""
    return ScannerModelInfo(
        manufacturer="TestVendor",
        model="UltraScan 9000",
        modality="CT",
        typical_noise_std=6.0,
        typical_uniformity=1.5,
        tube_types=["CeramicMax", "TungstenPro"],
        detector_type="PhotonCount-X",
        max_rotation_speed=0.20,
        year_introduced=2025,
        notes="Fictional scanner for unit testing.",
    )


@pytest.fixture()
def sample_instance(registry: ScannerRegistry) -> ScannerInstance:
    """A ScannerInstance backed by the Siemens SOMATOM Force built-in."""
    force = registry.get_model("Siemens", "SOMATOM Force")
    assert force is not None
    return ScannerInstance(
        scanner_id="TEST-CT-01",
        model_info=force,
        site_name="City General Hospital",
        location="Building B, Room 210",
        installation_date="2023-06-15",
        last_service_date="2025-11-01",
        custom_params={"local_noise_baseline": 7.2, "custom_protocol": "body_low_dose"},
    )


# ---------------------------------------------------------------------------
# 1. Built-in model count
# ---------------------------------------------------------------------------

def test_builtin_models_count(registry: ScannerRegistry) -> None:
    """The registry ships with exactly 5 built-in scanner models."""
    models = registry.list_models()
    assert len(models) == 5


# ---------------------------------------------------------------------------
# 2. Retrieve the Siemens SOMATOM Force by exact name
# ---------------------------------------------------------------------------

def test_get_siemens_force(registry: ScannerRegistry) -> None:
    """Look up the Siemens SOMATOM Force and verify key fields."""
    force = registry.get_model("Siemens", "SOMATOM Force")
    assert force is not None
    assert force.manufacturer == "Siemens"
    assert force.model == "SOMATOM Force"
    assert force.modality == "CT"
    assert force.typical_noise_std == 7.5
    assert force.typical_uniformity == 2.0
    assert force.tube_types == ["Vectron"]
    assert force.detector_type == "Stellar Infinity (dual-source)"
    assert force.max_rotation_speed == 0.25
    assert force.year_introduced == 2014


# ---------------------------------------------------------------------------
# 3. Case-insensitive model lookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "manufacturer, model",
    [
        ("siemens", "somatom force"),
        ("SIEMENS", "SOMATOM FORCE"),
        ("SiEmEnS", "SoMaToM fOrCe"),
        ("siemens", "SOMATOM Force"),
        ("Siemens", "somatom force"),
    ],
)
def test_get_model_case_insensitive(
    registry: ScannerRegistry,
    manufacturer: str,
    model: str,
) -> None:
    """get_model must match regardless of letter casing."""
    result = registry.get_model(manufacturer, model)
    assert result is not None
    assert result.manufacturer == "Siemens"
    assert result.model == "SOMATOM Force"


# ---------------------------------------------------------------------------
# 4. Unknown model returns None
# ---------------------------------------------------------------------------

def test_get_model_not_found_returns_none(registry: ScannerRegistry) -> None:
    """A non-existent manufacturer/model combination returns None."""
    assert registry.get_model("FakeVendor", "NoSuchScanner") is None


# ---------------------------------------------------------------------------
# 5. list_models returns all built-ins
# ---------------------------------------------------------------------------

def test_list_models_returns_all_builtins(registry: ScannerRegistry) -> None:
    """list_models should contain all 5 known manufacturers/models."""
    models = registry.list_models()
    names = {(m.manufacturer, m.model) for m in models}
    expected = {
        ("Siemens", "SOMATOM Force"),
        ("Siemens", "SOMATOM Definition AS+"),
        ("GE", "Revolution CT"),
        ("Philips", "iCT 256"),
        ("Canon", "Aquilion ONE"),
    }
    assert names == expected


# ---------------------------------------------------------------------------
# 6. Register and retrieve a scanner instance
# ---------------------------------------------------------------------------

def test_register_and_retrieve_instance(
    registry: ScannerRegistry,
    sample_instance: ScannerInstance,
) -> None:
    """After registering an instance, it should be retrievable by ID."""
    registry.register_instance(sample_instance)
    retrieved = registry.get_instance("TEST-CT-01")
    assert retrieved is not None
    assert retrieved.scanner_id == "TEST-CT-01"
    assert retrieved.site_name == "City General Hospital"
    assert retrieved.location == "Building B, Room 210"
    assert retrieved.installation_date == "2023-06-15"
    assert retrieved.last_service_date == "2025-11-01"
    assert retrieved.model_info.manufacturer == "Siemens"
    assert retrieved.model_info.model == "SOMATOM Force"


# ---------------------------------------------------------------------------
# 7. get_instance for unknown ID returns None
# ---------------------------------------------------------------------------

def test_get_instance_not_found_returns_none(registry: ScannerRegistry) -> None:
    """Querying a scanner_id that was never registered returns None."""
    assert registry.get_instance("DOES-NOT-EXIST") is None


# ---------------------------------------------------------------------------
# 8. list_instances is empty on a fresh registry
# ---------------------------------------------------------------------------

def test_list_instances_empty_initially(registry: ScannerRegistry) -> None:
    """No instances exist right after construction."""
    assert registry.list_instances() == []


# ---------------------------------------------------------------------------
# 9. register_model replaces an existing model with the same key
# ---------------------------------------------------------------------------

def test_register_model_replaces_existing(registry: ScannerRegistry) -> None:
    """Registering a model with an existing (manufacturer, model) pair replaces it."""
    replacement = ScannerModelInfo(
        manufacturer="Siemens",
        model="SOMATOM Force",
        modality="CT",
        typical_noise_std=6.5,  # updated noise
        typical_uniformity=1.8,
        tube_types=["Vectron-II"],
        detector_type="Next-gen detector",
        max_rotation_speed=0.20,
        year_introduced=2024,
        notes="Updated SOMATOM Force entry for testing.",
    )
    registry.register_model(replacement)

    # Total count should stay at 5 (replaced, not appended).
    assert len(registry.list_models()) == 5

    updated = registry.get_model("Siemens", "SOMATOM Force")
    assert updated is not None
    assert updated.typical_noise_std == 6.5
    assert updated.tube_types == ["Vectron-II"]
    assert updated.year_introduced == 2024


# ---------------------------------------------------------------------------
# 10. register_model adds a genuinely new model
# ---------------------------------------------------------------------------

def test_register_model_adds_new(
    registry: ScannerRegistry,
    sample_model_info: ScannerModelInfo,
) -> None:
    """A new manufacturer/model pair is appended, increasing the count."""
    registry.register_model(sample_model_info)
    assert len(registry.list_models()) == 6

    result = registry.get_model("TestVendor", "UltraScan 9000")
    assert result is not None
    assert result.typical_noise_std == 6.0


# ---------------------------------------------------------------------------
# 11. Verify detailed fields across all built-in ScannerModelInfo objects
# ---------------------------------------------------------------------------

_EXPECTED_NOISE = {
    ("Siemens", "SOMATOM Force"): 7.5,
    ("Siemens", "SOMATOM Definition AS+"): 8.0,
    ("GE", "Revolution CT"): 8.5,
    ("Philips", "iCT 256"): 9.0,
    ("Canon", "Aquilion ONE"): 8.0,
}

_EXPECTED_TUBE_TYPES = {
    ("Siemens", "SOMATOM Force"): ["Vectron"],
    ("Siemens", "SOMATOM Definition AS+"): ["Straton"],
    ("GE", "Revolution CT"): ["Performix HD"],
    ("Philips", "iCT 256"): ["iMRC"],
    ("Canon", "Aquilion ONE"): ["Megacool"],
}


@pytest.mark.parametrize(
    "manufacturer, model_name",
    list(_EXPECTED_NOISE.keys()),
    ids=[f"{m}-{n}" for m, n in _EXPECTED_NOISE],
)
def test_scanner_model_info_fields(
    registry: ScannerRegistry,
    manufacturer: str,
    model_name: str,
) -> None:
    """Each built-in model must have the correct noise_std and tube_types."""
    info = registry.get_model(manufacturer, model_name)
    assert info is not None
    assert info.typical_noise_std == _EXPECTED_NOISE[(manufacturer, model_name)]
    assert info.tube_types == _EXPECTED_TUBE_TYPES[(manufacturer, model_name)]
    assert info.modality == "CT"  # all built-ins are CT


# ---------------------------------------------------------------------------
# 12. ScannerInstance custom_params round-trip
# ---------------------------------------------------------------------------

def test_scanner_instance_custom_params(
    registry: ScannerRegistry,
    sample_instance: ScannerInstance,
) -> None:
    """custom_params should survive registration and retrieval intact."""
    registry.register_instance(sample_instance)
    retrieved = registry.get_instance("TEST-CT-01")
    assert retrieved is not None
    assert retrieved.custom_params == {
        "local_noise_baseline": 7.2,
        "custom_protocol": "body_low_dose",
    }


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_list_models_returns_copy(registry: ScannerRegistry) -> None:
    """list_models should return a new list; mutating it must not affect the registry."""
    models = registry.list_models()
    original_count = len(models)
    models.clear()
    assert len(registry.list_models()) == original_count


def test_register_instance_overwrites_same_id(registry: ScannerRegistry) -> None:
    """Registering a second instance with the same scanner_id overwrites the first."""
    force = registry.get_model("Siemens", "SOMATOM Force")
    assert force is not None

    inst_v1 = ScannerInstance(
        scanner_id="OVERWRITE-01",
        model_info=force,
        site_name="Hospital A",
        location="Room 1",
    )
    inst_v2 = ScannerInstance(
        scanner_id="OVERWRITE-01",
        model_info=force,
        site_name="Hospital B",
        location="Room 99",
    )

    registry.register_instance(inst_v1)
    registry.register_instance(inst_v2)

    result = registry.get_instance("OVERWRITE-01")
    assert result is not None
    assert result.site_name == "Hospital B"
    assert result.location == "Room 99"
    # Only one instance should exist, not two.
    assert len(registry.list_instances()) == 1


def test_register_model_case_insensitive_replacement(
    registry: ScannerRegistry,
) -> None:
    """register_model should detect duplicates case-insensitively."""
    replacement = ScannerModelInfo(
        manufacturer="ge",
        model="revolution ct",
        modality="CT",
        typical_noise_std=7.0,
        notes="Lower-cased replacement.",
    )
    registry.register_model(replacement)
    # Should still be 5 models total (replaced, not added).
    assert len(registry.list_models()) == 5
    # The stored object should be the new one with noise 7.0.
    result = registry.get_model("GE", "Revolution CT")
    assert result is not None
    assert result.typical_noise_std == 7.0


def test_scanner_model_info_defaults() -> None:
    """Verify that ScannerModelInfo optional fields default correctly."""
    minimal = ScannerModelInfo(
        manufacturer="MinVendor",
        model="MinModel",
        modality="PET_CT",
    )
    assert minimal.typical_noise_std is None
    assert minimal.typical_uniformity is None
    assert minimal.tube_types == []
    assert minimal.detector_type is None
    assert minimal.max_rotation_speed is None
    assert minimal.year_introduced is None
    assert minimal.notes == ""


def test_scanner_instance_defaults() -> None:
    """Verify that ScannerInstance optional fields default correctly."""
    model = ScannerModelInfo(
        manufacturer="V",
        model="M",
        modality="SPECT_CT",
    )
    inst = ScannerInstance(scanner_id="DEF-01", model_info=model)
    assert inst.site_name == ""
    assert inst.location == ""
    assert inst.installation_date is None
    assert inst.last_service_date is None
    assert inst.custom_params == {}


def test_multiple_instances_listed(registry: ScannerRegistry) -> None:
    """Registering N distinct instances should yield N entries in list_instances."""
    force = registry.get_model("Siemens", "SOMATOM Force")
    assert force is not None

    for i in range(4):
        inst = ScannerInstance(
            scanner_id=f"MULTI-{i:02d}",
            model_info=force,
            site_name=f"Site {i}",
        )
        registry.register_instance(inst)

    assert len(registry.list_instances()) == 4
    ids = {inst.scanner_id for inst in registry.list_instances()}
    assert ids == {"MULTI-00", "MULTI-01", "MULTI-02", "MULTI-03"}
