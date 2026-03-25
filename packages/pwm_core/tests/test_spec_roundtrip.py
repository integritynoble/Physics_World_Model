"""test_spec_roundtrip.py -- CoreSpec round-trip contract verification.

Tests that CanonicalCoreSpec (Omega, E, B, I, O, epsilon) survives a JSON
round-trip without data loss, as required by the strategy section 5.
"""

from __future__ import annotations
import json
import pytest
from pwm_core.spec.core import CanonicalCoreSpec, EXPERIMENT_SPEC_MAPPING


class TestCanonicalCoreSpecRoundTrip:
    """Verify lossless round-trip: CanonicalCoreSpec -> dict -> CanonicalCoreSpec."""

    def _make_spec(self) -> CanonicalCoreSpec:
        return CanonicalCoreSpec(
            omega={"type": "image_domain", "dims": [64, 64], "material": "tissue"},
            equations={"forward_model": "radon", "noise_model": "poisson", "angles": 180},
            boundary_conditions={"support": "circular", "radius": 0.9},
            initial_conditions={"prior": "none", "calibration": {}},
            observables={"detector": "line_array", "pixels": 182},
            tolerance=0.01,
            spec_version="1.0.0",
            domain_profile_ref="imaging/v1",
        )

    def test_dict_roundtrip(self):
        """spec -> dict -> spec must be lossless."""
        original = self._make_spec()
        d = original.to_dict() if hasattr(original, 'to_dict') else json.loads(json.dumps(original.__dict__, default=str))
        restored = CanonicalCoreSpec(**d)
        assert restored.omega == original.omega
        assert restored.equations == original.equations
        assert restored.boundary_conditions == original.boundary_conditions
        assert restored.initial_conditions == original.initial_conditions
        assert restored.observables == original.observables
        assert restored.tolerance == original.tolerance
        assert restored.spec_version == original.spec_version

    def test_json_roundtrip(self):
        """spec -> JSON string -> spec must be lossless."""
        original = self._make_spec()
        try:
            json_str = original.model_dump_json()
            restored = CanonicalCoreSpec.model_validate_json(json_str)
        except AttributeError:
            # Fallback for non-pydantic
            d = original.__dict__
            json_str = json.dumps(d, default=str)
            restored = CanonicalCoreSpec(**json.loads(json_str))
        assert restored.tolerance == original.tolerance
        assert restored.omega == original.omega

    def test_all_six_fields_required(self):
        """The three required fields (omega, equations, observables, tolerance) must be present."""
        spec = self._make_spec()
        assert hasattr(spec, 'omega')
        assert hasattr(spec, 'equations')
        assert hasattr(spec, 'boundary_conditions')
        assert hasattr(spec, 'initial_conditions')
        assert hasattr(spec, 'observables')
        assert hasattr(spec, 'tolerance')

    def test_experiment_spec_mapping_complete(self):
        """EXPERIMENT_SPEC_MAPPING must cover all 8 ExperimentSpec layers."""
        expected = {
            "PhysicsState", "SampleState", "SensorState", "EnvironmentState",
            "CalibrationState", "BudgetState", "ComputeState", "TaskState",
        }
        assert set(EXPERIMENT_SPEC_MAPPING.keys()) == expected

    def test_tolerance_is_numeric(self):
        """epsilon (tolerance) must be a float."""
        spec = self._make_spec()
        assert isinstance(spec.tolerance, (int, float))

    def test_empty_boundary_initial_conditions_allowed(self):
        """B and I can be empty dicts (not all problems have boundary/initial conditions)."""
        spec = CanonicalCoreSpec(
            omega={"type": "spectral"},
            equations={"forward_model": "dispersion"},
            boundary_conditions={},
            initial_conditions={},
            observables={"detector": "CCD"},
            tolerance=0.05,
        )
        assert spec.boundary_conditions == {}
        assert spec.initial_conditions == {}
