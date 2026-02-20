"""CasePack schema validation and version compatibility tests.

Tests verify that the ACR CT CasePack YAML can be loaded, parsed, and
validated correctly, and that required fields, metric sets, and ROI
definitions are complete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CASEPACKS_DIR = (
    Path(__file__).resolve().parents[2]
    / "pwm_core"
    / "clinical"
    / "casepacks"
)

_ACR_CT_YAML = _CASEPACKS_DIR / "acr_ct.yaml"

# ---------------------------------------------------------------------------
# Import guards for optional CasePack loader
# ---------------------------------------------------------------------------
try:
    from pwm_core.clinical.casepacks.casepack_loader import (
        CasePackConfig,
        list_available,
        load_casepack,
    )

    _LOADER_AVAILABLE = True
except ImportError:
    _LOADER_AVAILABLE = False


# ===================================================================
# Raw YAML loading tests (no loader dependency)
# ===================================================================


class TestLoadACRCTCasePackRaw:
    """Load acr_ct.yaml as raw YAML and verify structure."""

    def test_load_acr_ct_yaml_exists(self) -> None:
        """The acr_ct.yaml file should exist in the casepacks directory."""
        assert _ACR_CT_YAML.exists(), (
            f"acr_ct.yaml not found at {_ACR_CT_YAML}"
        )

    def test_load_acr_ct_yaml_valid(self) -> None:
        """The YAML should parse without errors and contain the 'casepack' key."""
        with open(_ACR_CT_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "casepack" in data, "Top-level 'casepack' key missing"

    def test_casepack_required_fields_raw(self) -> None:
        """The casepack dict should contain all required top-level fields.

        Required: id, name, version, min_pwm_version, phantom_type,
        series_selection, roi_definitions, metric_set, threshold_set.
        """
        with open(_ACR_CT_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cp = data["casepack"]

        required_fields = {
            "id", "name", "version", "min_pwm_version", "phantom_type",
            "series_selection", "roi_definitions", "metric_set",
            "threshold_set",
        }
        missing = required_fields - set(cp.keys())
        assert not missing, (
            f"CasePack missing required fields: {missing}"
        )

    def test_casepack_metric_set_count(self) -> None:
        """The metric_set should have exactly 12 entries per the ACR spec."""
        with open(_ACR_CT_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        metric_set = data["casepack"]["metric_set"]
        assert len(metric_set) == 12, (
            f"Expected 12 metrics, found {len(metric_set)}: {metric_set}"
        )

    def test_casepack_roi_definitions(self) -> None:
        """ROI definitions should include water_roi, peripheral_rois, insert_rois."""
        with open(_ACR_CT_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        roi_defs = data["casepack"]["roi_definitions"]

        expected_roi_groups = {"water_roi", "peripheral_rois", "insert_rois"}
        actual = set(roi_defs.keys())
        missing = expected_roi_groups - actual
        assert not missing, (
            f"ROI definitions missing groups: {missing}. Present: {actual}"
        )

    def test_casepack_insert_materials(self) -> None:
        """insert_rois should define bone, air, acrylic, polyethylene."""
        with open(_ACR_CT_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        insert_rois = data["casepack"]["roi_definitions"]["insert_rois"]

        expected_materials = {"bone", "air", "acrylic", "polyethylene"}
        actual = set(insert_rois.keys())
        missing = expected_materials - actual
        assert not missing, (
            f"Insert ROIs missing materials: {missing}"
        )


# ===================================================================
# Loader-based tests (skip if loader not implemented)
# ===================================================================


@pytest.mark.skipif(
    not _LOADER_AVAILABLE,
    reason="pwm_core.clinical.casepacks.casepack_loader not yet implemented",
)
class TestLoadACRCTCasePack:
    """Load acr_ct.yaml through the CasePack loader API."""

    def test_load_acr_ct_casepack(self) -> None:
        """load_casepack('acr_ct') should return a valid CasePackConfig."""
        config = load_casepack("acr_ct")
        assert isinstance(config, CasePackConfig)
        assert config.id == "acr_ct_v1.0"

    def test_casepack_required_fields(self) -> None:
        """A CasePackConfig with missing required fields should raise a
        validation error.
        """
        with pytest.raises((ValueError, TypeError, KeyError)):
            CasePackConfig(**{"name": "incomplete"})  # type: ignore[arg-type]

    def test_casepack_metric_set(self) -> None:
        """Loaded metric_set should have 12 entries."""
        config = load_casepack("acr_ct")
        assert len(config.metric_set) == 12

    def test_casepack_roi_definitions(self) -> None:
        """ROI defs should include water_roi, peripheral_rois, insert_rois."""
        config = load_casepack("acr_ct")
        roi_keys = set(config.roi_definitions.keys())
        expected = {"water_roi", "peripheral_rois", "insert_rois"}
        assert expected.issubset(roi_keys), (
            f"Missing ROI groups: {expected - roi_keys}"
        )

    def test_casepack_version_compat(self) -> None:
        """min_pwm_version should be a parseable version string."""
        config = load_casepack("acr_ct")
        parts = config.min_pwm_version.split(".")
        assert len(parts) >= 2, (
            f"min_pwm_version '{config.min_pwm_version}' not a valid semver"
        )

    def test_list_available(self) -> None:
        """list_available() should find at least the acr_ct casepack."""
        available = list_available()
        assert "acr_ct" in available, (
            f"acr_ct not in available casepacks: {available}"
        )
