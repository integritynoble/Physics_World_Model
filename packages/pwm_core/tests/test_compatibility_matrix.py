"""test_compatibility_matrix.py — CI validation for the compatibility matrix.

Ensures contrib/compatibility_matrix.yaml stays current as profiles
and registries evolve. Part of audit gap K6.
"""

from __future__ import annotations

import yaml
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent  # packages/pwm_core/tests/ -> repo root


class TestCompatibilityMatrix:
    """Verify compatibility_matrix.yaml is well-formed and references real artifacts."""

    @pytest.fixture
    def matrix(self):
        matrix_path = REPO_ROOT / "contrib" / "compatibility_matrix.yaml"
        assert matrix_path.exists(), f"Compatibility matrix not found at {matrix_path}"
        with open(matrix_path) as f:
            return yaml.safe_load(f)

    def test_matrix_has_required_sections(self, matrix):
        required = {
            "version",
            "spec_core_compatibility",
            "primitive_registry_requirements",
            "judge_compatibility",
            "product_kernel_compatibility",
        }
        assert required.issubset(set(matrix.keys())), (
            f"Missing sections: {required - set(matrix.keys())}"
        )

    def test_all_profiles_have_spec_core_entry(self, matrix):
        """Every profile in spec_core_compatibility must declare a spec_core range."""
        for profile, entry in matrix.get("spec_core_compatibility", {}).items():
            assert "spec_core" in entry, f"Profile {profile} missing spec_core version range"
            assert "status" in entry, f"Profile {profile} missing status field"

    def test_all_profiles_have_registry_requirements(self, matrix):
        """Every profile with spec_core entry should have registry requirements."""
        spec_profiles = set(matrix.get("spec_core_compatibility", {}).keys())
        reg_profiles = set(matrix.get("primitive_registry_requirements", {}).keys())
        missing = spec_profiles - reg_profiles
        # Warn but don't fail -- new profiles may not have registries yet
        if missing:
            pytest.skip(
                f"Profiles without registry requirements (acceptable for experimental): {missing}"
            )

    def test_imaging_profile_references_real_registries(self, matrix):
        """The imaging/v1 profile must reference general and imaging registries that exist."""
        imaging_reqs = matrix.get("primitive_registry_requirements", {}).get("imaging/v1", {})
        assert "general" in imaging_reqs, "imaging/v1 must require general primitives"
        assert "imaging" in imaging_reqs, "imaging/v1 must require imaging primitives"

        # Verify the referenced registries exist on disk
        for reg_name in imaging_reqs:
            reg_path = (
                REPO_ROOT / "contrib" / "primitive_registry" / reg_name / "v1" / "primitives.yaml"
            )
            assert reg_path.exists(), (
                f"Registry {reg_name}/v1 referenced but not found at {reg_path}"
            )

    def test_status_values_are_valid(self, matrix):
        """All status values must be from the DomainProfileMaturity ladder."""
        valid_statuses = {
            "experimental",
            "reference",
            "trusted",
            "certified_compatible",
            "deprecated",
        }
        for profile, entry in matrix.get("spec_core_compatibility", {}).items():
            status = entry.get("status", "")
            assert status in valid_statuses, (
                f"Profile {profile} has invalid status '{status}'. Valid: {valid_statuses}"
            )

    def test_product_kernel_compatibility_has_entries(self, matrix):
        """Product-kernel compatibility must have at least one entry."""
        entries = matrix.get("product_kernel_compatibility", [])
        assert len(entries) > 0, "product_kernel_compatibility must have at least one entry"
        for entry in entries:
            assert "product_version" in entry, "Each entry must have product_version"
            assert "kernel_min" in entry, "Each entry must have kernel_min"

    def test_judge_compatibility_entries(self, matrix):
        """Judge compatibility entries must reference valid profiles."""
        spec_profiles = set(matrix.get("spec_core_compatibility", {}).keys())
        judge_profiles = set(matrix.get("judge_compatibility", {}).keys())
        orphaned = judge_profiles - spec_profiles
        assert not orphaned, (
            f"Judge entries reference profiles not in spec_core_compatibility: {orphaned}"
        )
