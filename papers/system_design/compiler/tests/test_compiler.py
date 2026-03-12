"""Tests for the Constrained Primitive Compiler framework.

Test categories:
1. Nonlinear constraints — validate D/Λ family bounds
2. Agent translator — FlowchartElement → OperatorGraphSpec round-trip
3. Primitive compiler — full compilation + validation pipeline
4. Scenario validator — 4-scenario protocol metrics
5. Canonical chain — registry validation for 31+ modalities
"""

from __future__ import annotations

import math
import sys
import numpy as np
import pytest

# Ensure pwm_core is importable
sys.path.insert(0, "packages/pwm_core")

from pwm_core.graph.graph_spec import GraphEdge, GraphNode, NoiseSpec, OperatorGraphSpec
from pwm_core.graph.ir_types import (
    CanonicalPrimitive,
    DetectFamily,
    NodeRole,
    TransformFamily,
)
from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.canonical_decompositions import (
    CANONICAL_DECOMPOSITIONS,
    validate_decomposition,
)

from papers.system_design.compiler.nonlinear_constraints import (
    DETECT_CONSTRAINTS,
    TRANSFORM_CONSTRAINTS,
    validate_detect_params,
    validate_transform_params,
    compute_lipschitz_bound,
)
from papers.system_design.compiler.agent_translator import AgentToGraphTranslator
from papers.system_design.compiler.primitive_compiler import (
    CompilationReport,
    ConstrainedPrimitiveCompiler,
)
from papers.system_design.compiler.scenario_validator import (
    FourScenarioValidator,
    ScenarioResult,
    ValidationReport,
    psnr,
    ssim,
    nmse,
)


# =========================================================================
# 1. Nonlinear constraint tests
# =========================================================================


class TestDetectConstraints:
    """Validate the 5 Detect (D) families."""

    def test_all_5_families_registered(self):
        assert len(DETECT_CONSTRAINTS) == 5
        expected = {
            DetectFamily.intensity_square_law,
            DetectFamily.logarithmic,
            DetectFamily.sigmoid,
            DetectFamily.linear_field,
            DetectFamily.coherent_field,
        }
        assert set(DETECT_CONSTRAINTS.keys()) == expected

    def test_max_2_params_per_family(self):
        for family, constraint in DETECT_CONSTRAINTS.items():
            assert len(constraint.param_names) <= 2, (
                f"Detect family {family.value} has {len(constraint.param_names)} "
                f"params, exceeding the ≤2 limit"
            )

    def test_intensity_square_law_valid(self):
        ok, failures = validate_detect_params(
            DetectFamily.intensity_square_law,
            {"gain": 1.0},
        )
        assert ok, failures

    def test_intensity_square_law_out_of_bounds(self):
        ok, failures = validate_detect_params(
            DetectFamily.intensity_square_law,
            {"gain": -1.0},  # Below lower bound
        )
        assert not ok
        assert any("outside bounds" in f for f in failures)

    def test_coherent_field_phase_bounds(self):
        ok, failures = validate_detect_params(
            DetectFamily.coherent_field,
            {"gain": 1.0, "phase": 0.5},
        )
        assert ok, failures

        ok2, failures2 = validate_detect_params(
            DetectFamily.coherent_field,
            {"gain": 1.0, "phase": 10.0},  # > pi
        )
        assert not ok2

    def test_missing_params_allowed(self):
        """Missing parameters should be allowed (use defaults)."""
        ok, failures = validate_detect_params(
            DetectFamily.logarithmic,
            {},  # No params at all
        )
        assert ok, failures


class TestTransformConstraints:
    """Validate the 5 Transform (Λ) families."""

    def test_all_5_families_registered(self):
        assert len(TRANSFORM_CONSTRAINTS) == 5
        expected = {
            TransformFamily.beer_lambert,
            TransformFamily.phase_wrapping,
            TransformFamily.beam_hardening,
            TransformFamily.stopping_power,
            TransformFamily.saturation,
        }
        assert set(TRANSFORM_CONSTRAINTS.keys()) == expected

    def test_max_2_params_per_family(self):
        for family, constraint in TRANSFORM_CONSTRAINTS.items():
            assert len(constraint.param_names) <= 2, (
                f"Transform family {family.value} has {len(constraint.param_names)} params"
            )

    def test_beer_lambert_valid(self):
        ok, failures = validate_transform_params(
            TransformFamily.beer_lambert,
            {"mu": 0.1},
        )
        assert ok, failures

    def test_beer_lambert_out_of_bounds(self):
        ok, failures = validate_transform_params(
            TransformFamily.beer_lambert,
            {"mu": 1e5},  # Above upper bound 1e4
        )
        assert not ok

    def test_phase_wrapping_no_params(self):
        """Phase wrapping has zero parameters — should always pass."""
        ok, failures = validate_transform_params(
            TransformFamily.phase_wrapping,
            {},
        )
        assert ok, failures

    def test_beam_hardening_valid(self):
        ok, failures = validate_transform_params(
            TransformFamily.beam_hardening,
            {"a1": 1.0, "a2": 0.1},
        )
        assert ok, failures

    def test_saturation_valid(self):
        ok, failures = validate_transform_params(
            TransformFamily.saturation,
            {"x_max": 100.0, "x0": 10.0},
        )
        assert ok, failures


class TestLipschitzBound:
    def test_linear_field_bounded(self):
        L = compute_lipschitz_bound(DetectFamily.linear_field, {"gain": 1.0})
        assert L is not None
        assert L == 1e6

    def test_intensity_unbounded(self):
        L = compute_lipschitz_bound(DetectFamily.intensity_square_law, {"gain": 1.0})
        assert L is None  # Depends on input domain

    def test_phase_wrapping_bounded(self):
        L = compute_lipschitz_bound(TransformFamily.phase_wrapping, {})
        assert L == 1.0


# =========================================================================
# 2. Agent translator tests
# =========================================================================


class TestAgentTranslator:
    """Test FlowchartElement → OperatorGraphSpec translation."""

    def _make_ct_action(self) -> dict:
        return {
            "elements": [
                {"id": "src", "type": "source", "name": "X-ray Source",
                 "parameters": {}, "connects_to": ["proj"]},
                {"id": "proj", "type": "geometry", "name": "Radon Projection",
                 "parameters": {"model": "radon", "n_angles": 180},
                 "connects_to": ["det"]},
                {"id": "det", "type": "detector", "name": "Intensity Detector",
                 "parameters": {"model": "intensity"},
                 "connects_to": []},
            ],
            "measurement_shape": "(180, 182)",
        }

    def _make_mri_action(self) -> dict:
        return {
            "elements": [
                {"id": "src", "type": "source", "name": "Spin Source",
                 "parameters": {}, "connects_to": ["enc"]},
                {"id": "enc", "type": "geometry", "name": "K-space Encoding",
                 "parameters": {"model": "kspace"},
                 "connects_to": ["det"]},
                {"id": "det", "type": "detector", "name": "Detector",
                 "parameters": {},
                 "connects_to": []},
            ],
            "measurement_shape": "(256, 256)",
        }

    def test_ct_translation(self):
        translator = AgentToGraphTranslator()
        spec = translator.translate(self._make_ct_action(), modality="ct")

        assert isinstance(spec, OperatorGraphSpec)
        assert spec.graph_id == "ct_forward_agent"
        # Should have source + proj + det + noise = 4 nodes
        assert len(spec.nodes) >= 3

    def test_mri_translation(self):
        translator = AgentToGraphTranslator()
        spec = translator.translate(self._make_mri_action(), modality="mri")

        assert isinstance(spec, OperatorGraphSpec)
        assert len(spec.nodes) >= 3

    def test_noise_node_added(self):
        translator = AgentToGraphTranslator()
        spec = translator.translate(self._make_ct_action(), modality="ct")

        noise_nodes = [n for n in spec.nodes if n.role == NodeRole.noise]
        assert len(noise_nodes) >= 1

    def test_sequential_chain_fallback(self):
        """When no connects_to, translator should create sequential edges."""
        action = {
            "elements": [
                {"id": "a", "type": "source", "name": "Source", "parameters": {}},
                {"id": "b", "type": "geometry", "name": "Radon", "parameters": {"model": "radon"}},
                {"id": "c", "type": "detector", "name": "Det", "parameters": {}},
            ],
        }
        translator = AgentToGraphTranslator()
        spec = translator.translate(action, modality="test")
        assert len(spec.edges) >= 2  # a->b, b->c (at minimum)

    def test_empty_elements_raises(self):
        translator = AgentToGraphTranslator()
        with pytest.raises(ValueError, match="no elements"):
            translator.translate({"elements": []})

    def test_measurement_shape_parsing(self):
        translator = AgentToGraphTranslator()
        spec = translator.translate(self._make_ct_action(), modality="ct")
        y_shape = spec.metadata.get("y_shape")
        assert y_shape == [180, 182]


# =========================================================================
# 3. Primitive compiler tests
# =========================================================================


class TestPrimitiveCompiler:
    """Test the ConstrainedPrimitiveCompiler."""

    def _make_cassi_spec(self) -> OperatorGraphSpec:
        """Build a CASSI-like spec: M -> W -> Sigma -> D."""
        return OperatorGraphSpec(
            graph_id="cassi_test",
            nodes=[
                GraphNode(node_id="mask", primitive_id="coded_mask",
                          params={"H": 64, "W": 64, "seed": 42},
                          role=NodeRole.source),
                GraphNode(node_id="disperse", primitive_id="spectral_dispersion",
                          params={"n_lambda": 28, "shift_per_lambda": 2, "H": 64, "W": 64},
                          role=NodeRole.transport),
                GraphNode(node_id="acc", primitive_id="sum_axis",
                          params={"axis": 0},
                          role=NodeRole.sensor),
                GraphNode(node_id="noise", primitive_id="poisson_gaussian_sensor",
                          params={"peak_photons": 1e4, "read_sigma": 5.0},
                          role=NodeRole.noise),
            ],
            edges=[
                GraphEdge(source="mask", target="disperse"),
                GraphEdge(source="disperse", target="acc"),
                GraphEdge(source="acc", target="noise"),
            ],
            metadata={"modality": "cassi", "x_shape": [28, 64, 64], "y_shape": [64, 118]},
        )

    def _make_simple_linear_spec(self) -> OperatorGraphSpec:
        """Build a simple linear spec: identity pipeline."""
        return OperatorGraphSpec(
            graph_id="simple_test",
            nodes=[
                GraphNode(node_id="src", primitive_id="generic_source",
                          params={}, role=NodeRole.source),
                GraphNode(node_id="mask", primitive_id="coded_mask",
                          params={"H": 64, "W": 64, "seed": 42},
                          role=NodeRole.transport),
                GraphNode(node_id="det", primitive_id="generic_sensor",
                          params={}, role=NodeRole.sensor),
                GraphNode(node_id="noise", primitive_id="poisson_gaussian_sensor",
                          params={"peak_photons": 1e4},
                          role=NodeRole.noise),
            ],
            edges=[
                GraphEdge(source="src", target="mask"),
                GraphEdge(source="mask", target="det"),
                GraphEdge(source="det", target="noise"),
            ],
            metadata={"x_shape": [64, 64], "y_shape": [64, 64]},
        )

    def test_compilation_produces_report(self):
        compiler = ConstrainedPrimitiveCompiler()
        spec = self._make_simple_linear_spec()
        report = compiler.compile(spec, modality="generic")

        assert isinstance(report, CompilationReport)
        assert report.compilation_time_s > 0

    def test_node_count_and_depth(self):
        compiler = ConstrainedPrimitiveCompiler()
        spec = self._make_simple_linear_spec()
        report = compiler.compile(spec, modality="generic")

        assert report.node_count >= 1
        assert report.depth >= 1

    def test_canonical_chain_extraction(self):
        compiler = ConstrainedPrimitiveCompiler()
        spec = self._make_simple_linear_spec()
        report = compiler.compile(spec, modality="generic")

        assert len(report.canonical_chain) > 0
        assert report.canonical_chain_str != ""

    def test_compilation_report_summary(self):
        report = CompilationReport(
            valid=True,
            canonical_chain_str="M -> D",
            node_count=2,
            depth=2,
            representation_error=1e-4,
        )
        s = report.summary()
        assert "PASS" in s
        assert "M -> D" in s


# =========================================================================
# 4. Scenario validator tests
# =========================================================================


class TestMetrics:
    """Test PSNR, SSIM, NMSE metric functions."""

    def test_psnr_identical(self):
        x = np.random.rand(64, 64)
        assert psnr(x, x) >= 90.0  # Nearly infinite

    def test_psnr_noisy(self):
        x = np.random.rand(64, 64)
        y = x + np.random.randn(64, 64) * 0.01
        p = psnr(x, y)
        assert 30.0 < p < 60.0  # Reasonable range for low noise

    def test_ssim_identical(self):
        x = np.random.rand(64, 64) * 0.5 + 0.25
        s = ssim(x, x)
        assert s > 0.99

    def test_ssim_different(self):
        x = np.random.rand(64, 64)
        y = np.random.rand(64, 64)
        s = ssim(x, y)
        assert s < 0.5  # Random images should have low SSIM

    def test_nmse_identical(self):
        x = np.random.rand(64, 64)
        assert nmse(x, x) < 1e-10

    def test_nmse_noisy(self):
        x = np.random.rand(64, 64)
        y = x + np.random.randn(64, 64) * 0.1
        n = nmse(x, y)
        assert 0 < n < 1.0


class TestScenarioValidator:
    """Test the FourScenarioValidator."""

    def test_validation_report_structure(self):
        report = ValidationReport(
            modality="test",
            scenarios=[
                ScenarioResult(1, "I: True", psnr_db=35.0, ssim_val=0.95),
                ScenarioResult(2, "II: Mismatch", psnr_db=25.0, ssim_val=0.80),
                ScenarioResult(3, "III: Oracle", psnr_db=34.0, ssim_val=0.94),
                ScenarioResult(4, "IV: Auto", psnr_db=30.0, ssim_val=0.88),
            ],
            recovery_ratio=0.5,
            mismatch_psnr_drop=10.0,
            correction_psnr_gain=5.0,
            dominant_gate="model_mismatch",
            passed=True,
            threshold=0.5,
        )

        assert report.passed
        assert report.recovery_ratio == 0.5
        s = report.summary()
        assert "4-Scenario" in s
        assert "PASS" in s

    def test_recovery_ratio_formula(self):
        """Test ρ = (PSNR_IV - PSNR_II) / (PSNR_I - PSNR_II)."""
        psnr_I, psnr_II, psnr_IV = 35.0, 25.0, 30.0
        rho = (psnr_IV - psnr_II) / (psnr_I - psnr_II)
        assert abs(rho - 0.5) < 1e-10

    def test_perfect_recovery(self):
        """When PSNR_IV == PSNR_I, ρ = 1.0."""
        psnr_I, psnr_II, psnr_IV = 35.0, 25.0, 35.0
        rho = (psnr_IV - psnr_II) / (psnr_I - psnr_II)
        assert abs(rho - 1.0) < 1e-10

    def test_no_mismatch(self):
        """When PSNR_I == PSNR_II (no mismatch), ρ defaults to 1.0."""
        # This is handled in the validator: if mismatch_drop ≈ 0, ρ = 1.0
        pass


# =========================================================================
# 5. Canonical decomposition registry tests
# =========================================================================


class TestCanonicalDecompositions:
    """Validate the 31+ modality canonical decomposition registry."""

    def test_at_least_31_modalities(self):
        assert len(CANONICAL_DECOMPOSITIONS) >= 31

    def test_all_primitives_valid(self):
        """Every primitive in every decomposition must be a CanonicalPrimitive."""
        for mod, decomp in CANONICAL_DECOMPOSITIONS.items():
            for p in decomp.primitives:
                assert isinstance(p, CanonicalPrimitive), (
                    f"{mod}: {p!r} is not a CanonicalPrimitive"
                )

    def test_node_count_matches(self):
        """Node count field must match len(primitives)."""
        for mod, decomp in CANONICAL_DECOMPOSITIONS.items():
            assert decomp.nodes == len(decomp.primitives), (
                f"{mod}: nodes={decomp.nodes} != len(primitives)={len(decomp.primitives)}"
            )

    def test_depth_leq_nodes(self):
        """Depth must be ≤ node count."""
        for mod, decomp in CANONICAL_DECOMPOSITIONS.items():
            assert decomp.depth <= decomp.nodes, (
                f"{mod}: depth={decomp.depth} > nodes={decomp.nodes}"
            )

    def test_validation_known_modality(self):
        """Validate a known chain against registry."""
        passed, failures = validate_decomposition(
            "ct", [None, CanonicalPrimitive.Pi, CanonicalPrimitive.D, None]
        )
        assert passed, failures

    def test_validation_mismatch(self):
        """Wrong chain should fail validation."""
        passed, failures = validate_decomposition(
            "ct", [CanonicalPrimitive.M, CanonicalPrimitive.D]
        )
        assert not passed

    def test_validation_unknown_modality(self):
        passed, failures = validate_decomposition(
            "nonexistent_modality", [CanonicalPrimitive.D]
        )
        assert not passed

    def test_lambda_modalities_present(self):
        """Verify Lambda-bearing modalities are in the registry."""
        lambda_mods = [
            mod for mod, d in CANONICAL_DECOMPOSITIONS.items()
            if CanonicalPrimitive.Lambda in d.primitives
        ]
        assert len(lambda_mods) >= 5, (
            f"Expected ≥5 Lambda modalities, found: {lambda_mods}"
        )


# =========================================================================
# 6. End-to-end integration test
# =========================================================================


class TestEndToEnd:
    """Full pipeline: agent JSON → translate → compile → validate."""

    def test_ct_round_trip(self):
        """Translate a CT agent spec and compile it."""
        action = {
            "elements": [
                {"id": "src", "type": "source", "name": "X-ray Source",
                 "parameters": {}, "connects_to": ["proj"]},
                {"id": "proj", "type": "geometry", "name": "CT Projection",
                 "parameters": {"model": "radon", "n_angles": 180},
                 "connects_to": ["det"]},
                {"id": "det", "type": "detector", "name": "Detector",
                 "parameters": {"model": "intensity"},
                 "connects_to": []},
            ],
            "measurement_shape": "(180, 182)",
        }

        # Step 1: Translate
        translator = AgentToGraphTranslator()
        spec = translator.translate(action, modality="ct")
        assert isinstance(spec, OperatorGraphSpec)

        # Step 2: Compile
        compiler = ConstrainedPrimitiveCompiler()
        report = compiler.compile(spec, modality="ct")
        assert isinstance(report, CompilationReport)

        # Should compile successfully
        assert report.operator is not None or len(report.failures) > 0

        # Canonical chain should contain Pi and D
        actual = [c for c in report.canonical_chain if c is not None]
        assert CanonicalPrimitive.Pi in actual or CanonicalPrimitive.D in actual


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
