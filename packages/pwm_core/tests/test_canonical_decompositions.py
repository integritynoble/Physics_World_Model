"""Tests for canonical primitive decomposition infrastructure.

Validates the Finite Primitive Basis Theorem (FPB) integration:
- All implementation primitives have valid canonical mappings
- CANONICAL_REGISTRY is consistent with class attributes
- 31-modality decomposition registry is well-formed
- Extension protocol validation logic
- Basis growth curve saturation
"""

from __future__ import annotations

import pytest

from pwm_core.graph.ir_types import (
    CanonicalPrimitive,
    DetectFamily,
    PhysicsStageFamily,
)
from pwm_core.graph.primitives import (
    CANONICAL_REGISTRY,
    PRIMITIVE_REGISTRY,
    BasePrimitive,
    _ALL_PRIMITIVES,
    get_primitive,
)
from pwm_core.graph.canonical_decompositions import (
    CANONICAL_DECOMPOSITIONS,
    CanonicalDecomposition,
    N_MAX,
    validate_decomposition,
)
from pwm_core.graph.extension_protocol import (
    ExtensionProposal,
    basis_growth_curve,
    is_saturated,
    validate_extension,
)


# -----------------------------------------------------------------------
# Phase 1: Canonical enum + mapping tests
# -----------------------------------------------------------------------


class TestCanonicalEnums:
    """Test that canonical enums are well-formed."""

    def test_canonical_primitive_has_10_members(self):
        assert len(CanonicalPrimitive) == 10

    def test_canonical_primitive_names(self):
        expected = {"P", "M", "Pi", "F", "C", "Sigma", "D", "S", "W", "R"}
        assert set(cp.name for cp in CanonicalPrimitive) == expected

    def test_physics_stage_has_4_members(self):
        assert len(PhysicsStageFamily) == 4

    def test_detect_family_has_5_members(self):
        assert len(DetectFamily) == 5


class TestCanonicalMapping:
    """Test that all primitives have consistent canonical attributes."""

    def test_all_primitives_have_canonical_attrs(self):
        """Every primitive class has _canonical_id, _physics_stage, _detect_family."""
        for cls in _ALL_PRIMITIVES:
            assert hasattr(cls, '_canonical_id'), (
                f"{cls.__name__} missing _canonical_id"
            )
            assert hasattr(cls, '_physics_stage'), (
                f"{cls.__name__} missing _physics_stage"
            )
            assert hasattr(cls, '_detect_family'), (
                f"{cls.__name__} missing _detect_family"
            )

    def test_canonical_id_type(self):
        """_canonical_id is either None or a CanonicalPrimitive."""
        for cls in _ALL_PRIMITIVES:
            cid = cls._canonical_id
            assert cid is None or isinstance(cid, CanonicalPrimitive), (
                f"{cls.__name__}._canonical_id = {cid!r} is not valid"
            )

    def test_physics_stage_type(self):
        """_physics_stage is either None or a PhysicsStageFamily."""
        for cls in _ALL_PRIMITIVES:
            ps = cls._physics_stage
            assert ps is None or isinstance(ps, PhysicsStageFamily), (
                f"{cls.__name__}._physics_stage = {ps!r} is not valid"
            )

    def test_detect_family_type(self):
        """_detect_family is either None or a DetectFamily."""
        for cls in _ALL_PRIMITIVES:
            df = cls._detect_family
            assert df is None or isinstance(df, DetectFamily), (
                f"{cls.__name__}._detect_family = {df!r} is not valid"
            )

    def test_canonical_id_implies_physics_stage(self):
        """If _canonical_id is set, _physics_stage must also be set."""
        for cls in _ALL_PRIMITIVES:
            if cls._canonical_id is not None:
                assert cls._physics_stage is not None, (
                    f"{cls.__name__} has canonical_id={cls._canonical_id} "
                    f"but no physics_stage"
                )

    def test_detect_family_only_on_detect_primitives(self):
        """_detect_family should only be set when canonical_id is D."""
        for cls in _ALL_PRIMITIVES:
            if cls._detect_family is not None:
                assert cls._canonical_id == CanonicalPrimitive.D, (
                    f"{cls.__name__} has detect_family={cls._detect_family} "
                    f"but canonical_id={cls._canonical_id} (expected D)"
                )

    def test_physics_stage_consistency(self):
        """Canonical IDs map to correct physics stages."""
        stage_map = {
            CanonicalPrimitive.P: PhysicsStageFamily.propagation,
            CanonicalPrimitive.C: PhysicsStageFamily.propagation,
            CanonicalPrimitive.M: PhysicsStageFamily.interaction,
            CanonicalPrimitive.R: PhysicsStageFamily.interaction,
            CanonicalPrimitive.Pi: PhysicsStageFamily.encoding_projection,
            CanonicalPrimitive.F: PhysicsStageFamily.encoding_projection,
            CanonicalPrimitive.Sigma: PhysicsStageFamily.detection_readout,
            CanonicalPrimitive.D: PhysicsStageFamily.detection_readout,
            CanonicalPrimitive.S: PhysicsStageFamily.detection_readout,
            CanonicalPrimitive.W: PhysicsStageFamily.detection_readout,
        }
        for cls in _ALL_PRIMITIVES:
            cid = cls._canonical_id
            if cid is not None:
                expected_stage = stage_map[cid]
                assert cls._physics_stage == expected_stage, (
                    f"{cls.__name__}: canonical_id={cid.name} expects "
                    f"stage={expected_stage.value}, got {cls._physics_stage}"
                )


class TestCanonicalRegistry:
    """Test the CANONICAL_REGISTRY built at module load."""

    def test_registry_covers_all_10_types(self):
        """All 10 canonical primitives appear in the registry."""
        assert len(CANONICAL_REGISTRY) == 10
        for cp in CanonicalPrimitive:
            assert cp in CANONICAL_REGISTRY, (
                f"CanonicalPrimitive.{cp.name} missing from CANONICAL_REGISTRY"
            )

    def test_registry_values_are_valid_primitive_ids(self):
        """Every entry in CANONICAL_REGISTRY is a valid primitive_id."""
        for cp, pids in CANONICAL_REGISTRY.items():
            for pid in pids:
                assert pid in PRIMITIVE_REGISTRY, (
                    f"CANONICAL_REGISTRY[{cp.name}] contains unknown "
                    f"primitive_id '{pid}'"
                )

    def test_registry_consistent_with_class_attrs(self):
        """CANONICAL_REGISTRY matches class _canonical_id attributes."""
        for cls in _ALL_PRIMITIVES:
            cid = cls._canonical_id
            if cid is not None:
                assert cls.primitive_id in CANONICAL_REGISTRY[cid], (
                    f"{cls.__name__} ({cls.primitive_id}) has "
                    f"_canonical_id={cid.name} but is not in registry"
                )

    def test_no_duplicate_entries(self):
        """No primitive_id appears under multiple canonical types."""
        seen = {}
        for cp, pids in CANONICAL_REGISTRY.items():
            for pid in pids:
                assert pid not in seen, (
                    f"'{pid}' appears in both {seen[pid].name} and {cp.name}"
                )
                seen[pid] = cp


# -----------------------------------------------------------------------
# Phase 2: Canonical decomposition registry tests
# -----------------------------------------------------------------------


class TestCanonicalDecompositions:
    """Test the 31-modality decomposition registry."""

    def test_31_modalities_registered(self):
        assert len(CANONICAL_DECOMPOSITIONS) == 31

    def test_all_decompositions_well_formed(self):
        """Each decomposition has valid fields."""
        for key, decomp in CANONICAL_DECOMPOSITIONS.items():
            assert decomp.modality == key
            assert len(decomp.primitives) == decomp.nodes
            assert decomp.depth <= decomp.nodes
            assert decomp.depth > 0
            assert decomp.carrier in {
                "photon", "electron", "spin", "acoustic",
                "xray", "particle",
            }
            assert decomp.validation_level in {
                "full", "held_out", "exotic", "template",
            }

    def test_all_primitives_in_decompositions_are_canonical(self):
        """Every primitive in every decomposition is a CanonicalPrimitive."""
        for key, decomp in CANONICAL_DECOMPOSITIONS.items():
            for prim in decomp.primitives:
                assert isinstance(prim, CanonicalPrimitive), (
                    f"Decomposition '{key}' contains non-canonical "
                    f"primitive {prim!r}"
                )

    def test_node_count_within_bounds(self):
        """All decompositions have N <= N_MAX."""
        for key, decomp in CANONICAL_DECOMPOSITIONS.items():
            assert decomp.nodes <= N_MAX, (
                f"'{key}' has {decomp.nodes} nodes (max={N_MAX})"
            )

    def test_full_validation_modalities(self):
        """Check that the 7 full-validation modalities are present."""
        full = [
            k for k, d in CANONICAL_DECOMPOSITIONS.items()
            if d.validation_level == "full"
        ]
        assert len(full) == 7
        for m in ["cassi", "cacti", "spc", "lensless", "ptychography", "mri", "ct"]:
            assert m in full

    def test_dag_string_consistency(self):
        """The dag string matches the primitives list."""
        for key, decomp in CANONICAL_DECOMPOSITIONS.items():
            # Extract primitive names from dag string
            parts = [p.strip() for p in decomp.dag.replace("+", "->").split("->")]
            # Allow some flexibility in naming (D_coh -> D, etc.)
            clean_parts = [p.split("_")[0] for p in parts]
            # Strip special characters
            clean_parts = [
                p.replace("(", "").replace(")", "").strip()
                for p in clean_parts
                if p and p[0].isupper()
            ]
            assert len(clean_parts) == len(decomp.primitives), (
                f"'{key}': dag string '{decomp.dag}' has {len(clean_parts)} "
                f"primitives but primitives list has {len(decomp.primitives)}"
            )


# -----------------------------------------------------------------------
# Phase 4: Extension protocol tests
# -----------------------------------------------------------------------


class TestExtensionProtocol:
    """Test the 5-step extension protocol validation."""

    def test_valid_proposal_passes(self):
        proposal = ExtensionProposal(
            primitive_id="test_prim",
            canonical_name="Q",
            forward_adjoint_validated=True,
            representation_gap=0.05,
            error_with_primitive=0.001,
            modalities_requiring=["mod_a", "mod_b"],
            closure_test_passed=True,
        )
        passed, failures = validate_extension(proposal)
        assert passed
        assert len(failures) == 0

    def test_all_criteria_fail(self):
        proposal = ExtensionProposal(
            primitive_id="bad_prim",
            canonical_name="X",
        )
        passed, failures = validate_extension(proposal)
        assert not passed
        assert len(failures) == 5

    def test_step2_gap_too_small(self):
        proposal = ExtensionProposal(
            primitive_id="test",
            canonical_name="Q",
            forward_adjoint_validated=True,
            representation_gap=0.005,  # <= epsilon
            error_with_primitive=0.001,
            modalities_requiring=["a", "b"],
            closure_test_passed=True,
        )
        passed, failures = validate_extension(proposal)
        assert not passed
        assert any("Step 2" in f for f in failures)

    def test_step4_insufficient_modalities(self):
        proposal = ExtensionProposal(
            primitive_id="test",
            canonical_name="Q",
            forward_adjoint_validated=True,
            representation_gap=0.05,
            error_with_primitive=0.001,
            modalities_requiring=["only_one"],
            closure_test_passed=True,
        )
        passed, failures = validate_extension(proposal)
        assert not passed
        assert any("Step 4" in f for f in failures)


class TestBasisGrowth:
    """Test basis growth curve computation."""

    def test_curve_length_matches_modalities(self):
        curve = basis_growth_curve()
        assert len(curve) == len(CANONICAL_DECOMPOSITIONS)

    def test_curve_monotonically_nondecreasing(self):
        curve = basis_growth_curve()
        for i in range(1, len(curve)):
            assert curve[i][1] >= curve[i - 1][1]

    def test_curve_starts_at_1_or_more(self):
        curve = basis_growth_curve()
        assert curve[0][1] >= 1

    def test_saturation_at_10(self):
        """Final K should be exactly 10 (all canonical primitives used)."""
        curve = basis_growth_curve()
        assert curve[-1][1] == 10

    def test_is_saturated_returns_bool(self):
        result = is_saturated()
        assert isinstance(result, bool)


# -----------------------------------------------------------------------
# Phase 5: Compiler integration tests
# -----------------------------------------------------------------------


class TestCompilerCanonicalTags:
    """Test that GraphCompiler populates canonical tags in NodeTags."""

    def test_compiled_graph_has_canonical_tags(self):
        """Compile a simple graph and check NodeTags contain canonical fields."""
        from pwm_core.graph.graph_spec import (
            GraphEdge,
            GraphNode,
            OperatorGraphSpec,
        )
        from pwm_core.graph.compiler import GraphCompiler

        spec = OperatorGraphSpec(
            graph_id="test_canonical_tags",
            nodes=[
                GraphNode(
                    node_id="mask",
                    primitive_id="coded_mask",
                    params={"H": 16, "W": 16, "seed": 0},
                ),
                GraphNode(
                    node_id="sum",
                    primitive_id="sum_axis",
                    params={"axis": -1, "n": 8},
                ),
            ],
            edges=[
                GraphEdge(source="mask", target="sum"),
            ],
        )
        compiler = GraphCompiler()
        op = compiler.compile(spec, x_shape=(16, 16))

        # Check that canonical tags are populated
        mask_tags = op.node_tags["mask"]
        assert mask_tags.canonical_id == CanonicalPrimitive.M
        assert mask_tags.physics_stage == PhysicsStageFamily.interaction
        assert mask_tags.detect_family is None

        sum_tags = op.node_tags["sum"]
        assert sum_tags.canonical_id == CanonicalPrimitive.Sigma
        assert sum_tags.physics_stage == PhysicsStageFamily.detection_readout

    def test_to_canonical_method(self):
        """Test GraphOperator.to_canonical() and canonical_dag_string()."""
        from pwm_core.graph.graph_spec import (
            GraphEdge,
            GraphNode,
            OperatorGraphSpec,
        )
        from pwm_core.graph.compiler import GraphCompiler

        spec = OperatorGraphSpec(
            graph_id="test_to_canonical",
            nodes=[
                GraphNode(
                    node_id="mask",
                    primitive_id="coded_mask",
                    params={"H": 16, "W": 16, "seed": 0},
                ),
                GraphNode(
                    node_id="disperse",
                    primitive_id="spectral_dispersion",
                    params={"disp_step": 1},
                ),
                GraphNode(
                    node_id="integrate",
                    primitive_id="sum_axis",
                    params={"axis": -1, "n": 8},
                ),
            ],
            edges=[
                GraphEdge(source="mask", target="disperse"),
                GraphEdge(source="disperse", target="integrate"),
            ],
        )
        compiler = GraphCompiler()
        op = compiler.compile(spec, x_shape=(16, 16, 8))

        canonical = op.to_canonical()
        assert canonical == [
            CanonicalPrimitive.M,
            CanonicalPrimitive.W,
            CanonicalPrimitive.Sigma,
        ]
        assert op.canonical_dag_string() == "M -> W -> Sigma"
