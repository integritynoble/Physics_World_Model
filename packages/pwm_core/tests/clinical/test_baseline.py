"""Unit tests for CommissioningBundle management.

Tests cover creation, immutability, SHA-256 hashing, versioning,
active-baseline retrieval, delta comparison, and the full audit chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pwm_core.clinical.ct.baseline import (
    BaselineComparison,
    BaselineManager,
    CommissioningBundle,
)


# ===================================================================
# Helpers
# ===================================================================

_SAMPLE_METRICS: dict[str, dict[str, Any]] = {
    "ct_number_water": {"value": 0.5, "unit": "HU"},
    "noise_std": {"value": 7.5, "unit": "HU"},
    "uniformity": {"value": 2.0, "unit": "HU"},
    "geometric_accuracy": {"value": 200.0, "unit": "mm"},
}

_SAMPLE_OP_STATE: dict[str, Any] = {
    "kVp": 120,
    "mAs": 200,
    "kernel": "STANDARD",
}


def _create_manager(tmp_path: Path) -> BaselineManager:
    """Create a BaselineManager rooted at a temporary directory."""
    return BaselineManager(storage_path=tmp_path / "baselines")


def _create_bundle(
    mgr: BaselineManager,
    scanner_id: str = "CT-001",
    **kwargs: Any,
) -> CommissioningBundle:
    """Create a baseline bundle with sane defaults."""
    defaults: dict[str, Any] = {
        "scanner_model": "GE Revolution Apex",
        "metrics": _SAMPLE_METRICS,
        "operator_graph_state": _SAMPLE_OP_STATE,
        "approved_by": "J. Smith, MS, DABR",
        "service_event": "initial_installation",
        "casepack_id": "acr_ct_v1.0",
    }
    defaults.update(kwargs)
    return mgr.create_baseline(scanner_id=scanner_id, **defaults)


# ===================================================================
# Tests
# ===================================================================


class TestCreateBaseline:
    """Creating a baseline should populate all required fields."""

    def test_create_baseline(self, tmp_path: Path) -> None:
        """Create a baseline and verify all essential fields are present
        and contain plausible values.
        """
        mgr = _create_manager(tmp_path)
        bundle = _create_bundle(mgr)

        assert bundle.version == "1.0.0"
        assert bundle.scanner_id == "CT-001"
        assert bundle.scanner_model == "GE Revolution Apex"
        assert bundle.approved_by == "J. Smith, MS, DABR"
        assert bundle.service_event == "initial_installation"
        assert bundle.date  # non-empty ISO string
        assert bundle.metrics == _SAMPLE_METRICS
        assert bundle.operator_graph_state == _SAMPLE_OP_STATE
        assert bundle.sha256_inputs  # non-empty hex string
        assert bundle.sha256_outputs  # non-empty hex string
        assert bundle.provenance  # dict with at least 'casepack'
        assert bundle.previous_version is None


class TestBaselineImmutable:
    """CommissioningBundle is frozen; mutation should raise an error."""

    def test_baseline_immutable(self, tmp_path: Path) -> None:
        """Attempting to modify a frozen Pydantic model field should raise
        a ``ValidationError``.
        """
        mgr = _create_manager(tmp_path)
        bundle = _create_bundle(mgr)

        with pytest.raises(ValidationError):
            bundle.version = "2.0.0"  # type: ignore[misc]


class TestBaselineSHA256:
    """SHA-256 hashes should be computed and non-trivial."""

    def test_baseline_sha256(self, tmp_path: Path) -> None:
        """Both ``sha256_inputs`` and ``sha256_outputs`` should be 64-char
        hex strings (256 bits).
        """
        mgr = _create_manager(tmp_path)
        bundle = _create_bundle(mgr)

        assert len(bundle.sha256_inputs) == 64
        assert len(bundle.sha256_outputs) == 64
        # Distinct hashes
        assert bundle.sha256_inputs != bundle.sha256_outputs


class TestBaselineVersioning:
    """Creating successive baselines for the same scanner should auto-increment."""

    def test_baseline_versioning(self, tmp_path: Path) -> None:
        """Create v1.0.0 and then a second baseline; the second should be
        v1.0.1 with ``previous_version`` pointing back to v1.0.0.
        """
        mgr = _create_manager(tmp_path)
        b1 = _create_bundle(mgr)
        assert b1.version == "1.0.0"
        assert b1.previous_version is None

        b2 = _create_bundle(mgr, service_event="annual_qc")
        assert b2.version == "1.0.1"
        assert b2.previous_version == "1.0.0"


class TestGetActiveBaseline:
    """get_active_baseline should return the latest version."""

    def test_get_active_baseline(self, tmp_path: Path) -> None:
        """After creating two baselines, the active one should be the second."""
        mgr = _create_manager(tmp_path)
        _create_bundle(mgr)
        b2 = _create_bundle(mgr, service_event="tube_change")

        active = mgr.get_active_baseline("CT-001")
        assert active is not None
        assert active.version == b2.version


class TestCompareToBaseline:
    """compare_to_baseline should compute correct deltas."""

    def test_compare_to_baseline(self, tmp_path: Path) -> None:
        """A known delta should produce the correct BaselineComparison.

        Baseline water = 0.5 HU; current = 3.5 HU; delta = 3.0 HU.
        """
        mgr = _create_manager(tmp_path)
        bundle = _create_bundle(mgr)

        current: dict[str, float] = {
            "ct_number_water": 3.5,
            "noise_std": 7.5,
        }
        comparisons = mgr.compare_to_baseline(current, bundle)

        assert "ct_number_water" in comparisons
        cmp_water = comparisons["ct_number_water"]
        assert cmp_water.baseline_value == 0.5
        assert cmp_water.current_value == 3.5
        assert abs(cmp_water.delta - 3.0) < 1e-6

        # noise_std is unchanged
        cmp_noise = comparisons["noise_std"]
        assert cmp_noise.status == "STABLE"


class TestBaselineAuditChain:
    """Multiple versions should form an ordered audit chain."""

    def test_baseline_audit_chain(self, tmp_path: Path) -> None:
        """Create three baselines and verify list_baselines returns all
        three in version order.
        """
        mgr = _create_manager(tmp_path)
        b1 = _create_bundle(mgr)
        b2 = _create_bundle(mgr, service_event="tube_change")
        b3 = _create_bundle(mgr, service_event="software_upgrade")

        chain = mgr.list_baselines("CT-001")
        assert len(chain) == 3
        assert chain[0].version == "1.0.0"
        assert chain[1].version == "1.0.1"
        assert chain[2].version == "1.0.2"

        # Verify chaining
        assert chain[1].previous_version == "1.0.0"
        assert chain[2].previous_version == "1.0.1"
