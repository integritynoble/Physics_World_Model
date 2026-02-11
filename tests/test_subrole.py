"""Tests for PhysicsSubrole classification and carrier_transition enforcement (R2)."""

import pytest

from pwm_core.graph.ir_types import PhysicsSubrole, NodeTags
from pwm_core.graph.compiler import GraphCompilationError, GraphCompiler
from pwm_core.graph.graph_spec import GraphNode, OperatorGraphSpec
from pwm_core.graph.primitives import (
    PRIMITIVE_REGISTRY,
    BasePrimitive,
    FresnelProp,
    AngularSpectrum,
    RayTrace,
    Conv2d,
    Conv3d,
    DeconvRL,
    CodedMask,
    DMDPattern,
    SIMPattern,
    SpectralDispersion,
    ChromaticWarp,
    RandomMask,
    CTRadon,
    MRIKspace,
    TemporalMask,
    MagnitudeSq,
    Saturation,
    LogCompress,
    PoissonNoise,
    GaussianNoise,
    PoissonGaussianNoise,
    FPN,
    FrameIntegration,
    MotionWarp,
    Quantize,
    ADCClip,
    Identity,
    SumAxis,
    PhotonSource,
    XRaySource,
    AcousticSource,
    SpinSource,
    GenericSource,
    PhotonSensor,
    CoilSensor,
    TransducerSensor,
    GenericSensor,
    PoissonGaussianSensorNoise,
    ComplexGaussianSensorNoise,
    PoissonOnlySensorNoise,
    Interference,
    FourierRelay,
    MaxwellInterface,
)


# ---------------------------------------------------------------------------
# Test 1: PhysicsSubrole enum has all 7 values
# ---------------------------------------------------------------------------


class TestPhysicsSubroleEnum:
    def test_has_all_seven_values(self):
        expected = {
            "propagation", "modulation", "sampling",
            "interaction", "transduction", "encoding", "relay",
        }
        actual = {member.value for member in PhysicsSubrole}
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_is_string_enum(self):
        assert isinstance(PhysicsSubrole.propagation, str)
        assert PhysicsSubrole.propagation == "propagation"


# ---------------------------------------------------------------------------
# Test 2: All primitives have correct _physics_subrole
# ---------------------------------------------------------------------------


class TestPrimitiveSubroles:
    """Verify that every primitive has the correct _physics_subrole class attribute."""

    # Map: primitive class -> expected subrole string (or None)
    EXPECTED = {
        # Propagation family
        FresnelProp: "propagation",
        AngularSpectrum: "propagation",
        RayTrace: "propagation",
        Conv2d: "propagation",
        Conv3d: "propagation",
        DeconvRL: "propagation",
        ChromaticWarp: "propagation",
        MotionWarp: "propagation",
        MaxwellInterface: "propagation",
        # Modulation family
        CodedMask: "modulation",
        DMDPattern: "modulation",
        SIMPattern: "modulation",
        # Sampling family
        RandomMask: "sampling",
        CTRadon: "sampling",
        MRIKspace: "sampling",
        # Encoding family
        SpectralDispersion: "encoding",
        TemporalMask: "encoding",
        FrameIntegration: "encoding",
        SumAxis: "encoding",
        # Transduction family
        MagnitudeSq: "transduction",
        Saturation: "transduction",
        LogCompress: "transduction",
        Quantize: "transduction",
        ADCClip: "transduction",
        # Interaction family
        Interference: "interaction",
        # Relay family
        Identity: "relay",
        FourierRelay: "relay",
        # Noise (None)
        PoissonNoise: None,
        GaussianNoise: None,
        PoissonGaussianNoise: None,
        FPN: None,
        # Source (None)
        PhotonSource: None,
        XRaySource: None,
        AcousticSource: None,
        SpinSource: None,
        GenericSource: None,
        # Sensor (None)
        PhotonSensor: None,
        CoilSensor: None,
        TransducerSensor: None,
        GenericSensor: None,
        # Sensor noise (None)
        PoissonGaussianSensorNoise: None,
        ComplexGaussianSensorNoise: None,
        PoissonOnlySensorNoise: None,
    }

    def test_all_subroles_correct(self):
        for cls, expected_sr in self.EXPECTED.items():
            actual = getattr(cls, "_physics_subrole", None)
            assert actual == expected_sr, (
                f"{cls.__name__}._physics_subrole = {actual!r}, expected {expected_sr!r}"
            )

    def test_base_primitive_has_none(self):
        assert BasePrimitive._physics_subrole is None


# ---------------------------------------------------------------------------
# Test 3: Compiled graph has physics_subrole populated in NodeTags
# ---------------------------------------------------------------------------


class TestCompiledSubroleTags:
    def test_subrole_in_compiled_tags(self):
        """Compile a simple graph and verify NodeTags.physics_subrole is populated."""
        compiler = GraphCompiler()
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_subrole_tags",
            "metadata": {"x_shape": [64, 64], "y_shape": [64, 64]},
            "nodes": [
                {"node_id": "blur", "primitive_id": "conv2d",
                 "params": {"sigma": 2.0, "mode": "reflect"}},
                {"node_id": "mask", "primitive_id": "coded_mask",
                 "params": {"seed": 42, "H": 64, "W": 64}},
                {"node_id": "ident", "primitive_id": "identity",
                 "params": {}},
            ],
            "edges": [
                {"source": "blur", "target": "mask"},
                {"source": "mask", "target": "ident"},
            ],
        })
        graph_op = compiler.compile(spec)

        assert graph_op.node_tags["blur"].physics_subrole == PhysicsSubrole.propagation
        assert graph_op.node_tags["mask"].physics_subrole == PhysicsSubrole.modulation
        assert graph_op.node_tags["ident"].physics_subrole == PhysicsSubrole.relay

    def test_noise_primitive_subrole_is_none(self):
        """Noise primitives should have physics_subrole=None in compiled tags."""
        compiler = GraphCompiler()
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_noise_subrole",
            "metadata": {"x_shape": [64, 64], "y_shape": [64, 64]},
            "nodes": [
                {"node_id": "n", "primitive_id": "poisson_gaussian_sensor",
                 "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [],
        })
        graph_op = compiler.compile(spec)
        assert graph_op.node_tags["n"].physics_subrole is None


# ---------------------------------------------------------------------------
# Test 4: Photoacoustic-like graph with carrier_transitions + interaction node -> passes
# ---------------------------------------------------------------------------


class TestCarrierTransitions:
    def _pa_graph(self, include_interaction: bool):
        """Build a photoacoustic-like canonical graph."""
        nodes = [
            {"node_id": "source", "primitive_id": "photon_source",
             "role": "source", "params": {"strength": 1.0}},
            {"node_id": "prop", "primitive_id": "conv2d",
             "role": "transport", "params": {"sigma": 2.0, "mode": "reflect"}},
        ]
        edges = [
            {"source": "source", "target": "prop"},
        ]
        if include_interaction:
            nodes.append({
                "node_id": "pa_convert", "primitive_id": "magnitude_sq",
                "role": "interaction", "params": {},
            })
            edges.append({"source": "prop", "target": "pa_convert"})
            next_src = "pa_convert"
        else:
            next_src = "prop"

        nodes.extend([
            {"node_id": "sensor", "primitive_id": "transducer_sensor",
             "role": "sensor", "params": {"sensitivity": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
             "role": "noise", "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
        ])
        edges.extend([
            {"source": next_src, "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ])
        return {
            "graph_id": "test_pa",
            "metadata": {
                "canonical_chain": True,
                "carrier_transitions": ["photon->acoustic"],
                "x_shape": [64, 64],
                "y_shape": [64, 64],
            },
            "nodes": nodes,
            "edges": edges,
        }

    def test_with_interaction_node_passes(self):
        """Graph with carrier_transitions and an interaction/transduction node should pass."""
        compiler = GraphCompiler()
        spec = OperatorGraphSpec.model_validate(self._pa_graph(include_interaction=True))
        # Should not raise
        graph_op = compiler.compile(spec)
        assert graph_op is not None

    # -----------------------------------------------------------------------
    # Test 5: carrier_transitions WITHOUT interaction node -> raises
    # -----------------------------------------------------------------------

    def test_without_interaction_node_raises(self):
        """Graph with carrier_transitions but NO interaction/transduction node should fail."""
        compiler = GraphCompiler()
        spec = OperatorGraphSpec.model_validate(self._pa_graph(include_interaction=False))
        with pytest.raises(GraphCompilationError, match="carrier_transitions"):
            compiler.compile(spec)


# ---------------------------------------------------------------------------
# Test 6: All existing graphs compile without carrier_transitions (backward compat)
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_simple_graph_without_carrier_transitions(self):
        """Graphs without carrier_transitions should compile as before."""
        compiler = GraphCompiler()
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_compat_no_ct",
            "metadata": {
                "canonical_chain": True,
                "x_shape": [64, 64],
                "y_shape": [64, 64],
            },
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport", "params": {"sigma": 2.0, "mode": "reflect"}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor", "params": {"quantum_efficiency": 0.9}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise",
                 "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "blur"},
                {"source": "blur", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
            ],
        })
        graph_op = compiler.compile(spec)
        assert graph_op is not None

    def test_non_canonical_graph_ignores_carrier_transitions(self):
        """Non-canonical graphs should not trigger carrier_transition checks."""
        compiler = GraphCompiler()
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_compat_non_canon",
            "metadata": {"x_shape": [32, 32], "y_shape": [32, 32]},
            "nodes": [
                {"node_id": "blur", "primitive_id": "conv2d",
                 "params": {"sigma": 2.0, "mode": "reflect"}},
            ],
            "edges": [],
        })
        graph_op = compiler.compile(spec)
        assert graph_op is not None


# ---------------------------------------------------------------------------
# Test 7: GraphNode accepts physics_subrole field
# ---------------------------------------------------------------------------


class TestGraphNodeSubrole:
    def test_accepts_physics_subrole(self):
        node = GraphNode(
            node_id="test",
            primitive_id="conv2d",
            physics_subrole=PhysicsSubrole.propagation,
        )
        assert node.physics_subrole == PhysicsSubrole.propagation

    def test_defaults_to_none(self):
        node = GraphNode(
            node_id="test",
            primitive_id="conv2d",
        )
        assert node.physics_subrole is None

    def test_physics_subrole_in_node_tags(self):
        tags = NodeTags(physics_subrole=PhysicsSubrole.modulation)
        assert tags.physics_subrole == PhysicsSubrole.modulation

    def test_node_tags_subrole_defaults_none(self):
        tags = NodeTags()
        assert tags.physics_subrole is None
