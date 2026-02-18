"""test_v4plus_primitives.py
==============================

Tests for Phase v4+ state extensions (5 new state classes) and
~27 new primitives (23 element + 4 sensor).

Covers:
- State class creation, field access, and factory function
- Registry presence for all new primitives
- forward() shape and sanity (finite, non-zero for non-zero input)
- Adjoint existence for linear primitives
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# State imports
# ---------------------------------------------------------------------------
from pwm_core.graph.state_spec import (
    EventListState,
    KSpaceState,
    PhaseSpaceState,
    RayBundleState,
    StateSpec,
    WaveFieldState,
    make_state_spec,
)
from pwm_core.graph.ir_types import CarrierType

# ---------------------------------------------------------------------------
# Primitive imports
# ---------------------------------------------------------------------------
from pwm_core.graph.primitives import (
    PRIMITIVE_REGISTRY,
    get_primitive,
    # Element primitives
    ScanTrajectory,
    TimeOfFlightGate,
    CollimatorModel,
    FluoroTemporalIntegrator,
    FluorescenceKinetics,
    NonlinearExcitation,
    SaturationDepletion,
    BlinkingEmitterModel,
    EvanescentFieldDecay,
    DopplerEstimator,
    ElasticWaveModel,
    DualEnergyBeerLambert,
    DepthOptics,
    DiffractionCamera,
    SARBackprojection,
    ParticleAttenuation,
    MultipleScatteringKernel,
    VesselFlowContrast,
    SpecularReflectionModel,
    StructuredLightProjector,
    SequenceBlock,
    PhysiologyDrift,
    ReciprocalSpaceGeometry,
    # Sensor primitives
    SPADToFSensor,
    EnergyResolvingDetector,
    FiberBundleSensor,
    TrackDetectorSensor,
)


# =========================================================================
# Task 1: State extension tests
# =========================================================================


class TestWaveFieldState:
    """Tests for WaveFieldState."""

    def test_creation_defaults(self):
        s = WaveFieldState()
        assert s.carrier_type == CarrierType.photon
        assert s.representation == "complex_field"
        assert s.wavelength_nm == 550.0
        assert s.frequency_hz is None

    def test_custom_fields(self):
        s = WaveFieldState(wavelength_nm=633.0, frequency_hz=4.74e14)
        assert s.wavelength_nm == 633.0
        assert s.frequency_hz == 4.74e14

    def test_in_union(self):
        """WaveFieldState is a valid StateSpec member."""
        s: StateSpec = WaveFieldState()
        assert s.carrier_type == CarrierType.photon


class TestRayBundleState:
    """Tests for RayBundleState."""

    def test_creation_defaults(self):
        s = RayBundleState()
        assert s.carrier_type == CarrierType.photon
        assert s.representation == "ray_bundle"
        assert s.n_rays == 1024
        assert s.max_bounces == 1

    def test_custom_fields(self):
        s = RayBundleState(n_rays=4096, max_bounces=5)
        assert s.n_rays == 4096
        assert s.max_bounces == 5


class TestEventListState:
    """Tests for EventListState."""

    def test_creation_defaults(self):
        s = EventListState()
        assert s.carrier_type == CarrierType.photon
        assert s.representation == "event_list"
        assert s.time_resolved is True
        assert s.bin_width_ns == 1.0

    def test_custom_fields(self):
        s = EventListState(time_resolved=False, bin_width_ns=0.5)
        assert s.time_resolved is False
        assert s.bin_width_ns == 0.5


class TestKSpaceState:
    """Tests for KSpaceState."""

    def test_creation_defaults(self):
        s = KSpaceState()
        assert s.carrier_type == CarrierType.spin
        assert s.representation == "kspace"
        assert s.trajectory_type == "cartesian"
        assert s.undersampling_factor == 1.0

    def test_custom_fields(self):
        s = KSpaceState(trajectory_type="spiral", undersampling_factor=4.0)
        assert s.trajectory_type == "spiral"
        assert s.undersampling_factor == 4.0


class TestPhaseSpaceState:
    """Tests for PhaseSpaceState."""

    def test_creation_defaults(self):
        s = PhaseSpaceState()
        assert s.carrier_type == CarrierType.abstract
        assert s.representation == "phase_space"
        assert s.energy_kev == 1.0
        assert s.angular_spread_mrad == 0.0

    def test_custom_fields(self):
        s = PhaseSpaceState(energy_kev=100.0, angular_spread_mrad=5.0)
        assert s.energy_kev == 100.0
        assert s.angular_spread_mrad == 5.0


class TestMakeStateSpec:
    """Tests for the make_state_spec factory function."""

    def test_photon(self):
        s = make_state_spec(CarrierType.photon)
        assert s.carrier_type == CarrierType.photon

    def test_electron(self):
        s = make_state_spec(CarrierType.electron)
        assert s.carrier_type == CarrierType.electron

    def test_acoustic(self):
        s = make_state_spec(CarrierType.acoustic)
        assert s.carrier_type == CarrierType.acoustic

    def test_spin(self):
        s = make_state_spec(CarrierType.spin)
        assert s.carrier_type == CarrierType.spin

    def test_abstract_maps_to_phase_space(self):
        s = make_state_spec(CarrierType.abstract)
        assert isinstance(s, PhaseSpaceState)
        assert s.carrier_type == CarrierType.abstract

    def test_unknown_carrier_falls_back(self):
        """particle_other should fall back to PhotonState."""
        s = make_state_spec(CarrierType.particle_other)
        assert s.carrier_type == CarrierType.particle_other


# =========================================================================
# Task 2: Primitive registration tests
# =========================================================================

# All 27 new primitive IDs
NEW_PRIMITIVE_IDS = [
    # Element primitives
    "scan_trajectory",
    "tof_gate",
    "collimator_model",
    "fluoro_temporal_integrator",
    "fluorescence_kinetics",
    "nonlinear_excitation",
    "saturation_depletion",
    "blinking_emitter",
    "evanescent_decay",
    "doppler_estimator",
    "elastic_wave_model",
    "dual_energy_beer_lambert",
    "depth_optics",
    "diffraction_camera",
    "sar_backprojection",
    "particle_attenuation",
    "multiple_scattering",
    "vessel_flow_contrast",
    "specular_reflection",
    "structured_light_projector",
    "sequence_block",
    "physiology_drift",
    "reciprocal_space_geometry",
    # Sensor primitives
    "spad_tof_sensor",
    "energy_resolving_detector",
    "fiber_bundle_sensor",
    "track_detector_sensor",
]


class TestPrimitiveRegistration:
    """All new primitives must be in PRIMITIVE_REGISTRY."""

    @pytest.mark.parametrize("pid", NEW_PRIMITIVE_IDS)
    def test_registered(self, pid: str):
        assert pid in PRIMITIVE_REGISTRY, f"{pid} not in PRIMITIVE_REGISTRY"

    @pytest.mark.parametrize("pid", NEW_PRIMITIVE_IDS)
    def test_get_primitive(self, pid: str):
        prim = get_primitive(pid)
        assert prim.primitive_id == pid

    @pytest.mark.parametrize("pid", NEW_PRIMITIVE_IDS)
    def test_has_physics_tier(self, pid: str):
        cls = PRIMITIVE_REGISTRY[pid]
        assert hasattr(cls, "_physics_tier"), f"{pid} missing _physics_tier"
        assert cls._physics_tier is not None

    @pytest.mark.parametrize("pid", NEW_PRIMITIVE_IDS)
    def test_serialize(self, pid: str):
        prim = get_primitive(pid)
        s = prim.serialize()
        assert "primitive_id" in s
        assert s["primitive_id"] == pid
        assert "is_linear" in s


# =========================================================================
# Task 2: Element primitive forward tests
# =========================================================================


class TestScanTrajectory:
    def test_forward_shape(self):
        p = ScanTrajectory({"n_points": 16})
        x = np.random.rand(32, 32)
        y = p.forward(x)
        assert y.shape == (16,)
        assert np.all(np.isfinite(y))

    def test_adjoint_exists(self):
        p = ScanTrajectory({"n_points": 16, "x_shape": (32, 32)})
        y = np.random.rand(16)
        x_adj = p.adjoint(y)
        assert x_adj.shape == (32, 32)


class TestTimeOfFlightGate:
    def test_forward_2d(self):
        p = TimeOfFlightGate({"n_bins": 8, "bin_width_ns": 1.0})
        x = np.random.rand(32, 32)
        y = p.forward(x)
        assert y.ndim == 2
        assert np.all(np.isfinite(y))

    def test_forward_with_jitter(self):
        p = TimeOfFlightGate({"n_bins": 8, "timing_jitter_ns": 1.0})
        x = np.random.rand(32, 32)
        y = p.forward(x)
        assert np.all(np.isfinite(y))


class TestCollimatorModel:
    def test_forward_shape(self):
        p = CollimatorModel()
        x = np.random.rand(32, 32)
        y = p.forward(x)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))

    def test_nonzero_for_nonzero_input(self):
        p = CollimatorModel()
        x = np.ones((16, 16))
        y = p.forward(x)
        assert np.sum(np.abs(y)) > 0


class TestFluoroTemporalIntegrator:
    def test_forward_3d(self):
        p = FluoroTemporalIntegrator({"n_frames": 4})
        x = np.random.rand(4, 16, 16)
        y = p.forward(x)
        assert y.shape == (16, 16)

    def test_forward_2d_with_motion(self):
        p = FluoroTemporalIntegrator({"motion_sigma": 1.0})
        x = np.random.rand(16, 16)
        y = p.forward(x)
        assert y.shape == (16, 16)
        assert np.all(np.isfinite(y))


class TestFluorescenceKinetics:
    def test_forward_decays(self):
        p = FluorescenceKinetics({"lifetime_ns": 3.0})
        x = np.ones((16, 16))
        y = p.forward(x)
        assert np.all(y < x)  # decay reduces intensity
        assert np.all(np.isfinite(y))

    def test_is_nonlinear(self):
        p = FluorescenceKinetics()
        assert p.is_linear is False


class TestNonlinearExcitation:
    def test_two_photon(self):
        p = NonlinearExcitation({"n_photons": 2})
        x = np.array([[2.0, 3.0], [4.0, 5.0]])
        y = p.forward(x)
        np.testing.assert_allclose(y, np.array([[4.0, 9.0], [16.0, 25.0]]))

    def test_is_nonlinear(self):
        assert NonlinearExcitation._is_linear is False


class TestSaturationDepletion:
    def test_forward_shape(self):
        p = SaturationDepletion({"depletion_factor": 0.5, "psf_sigma": 2.0})
        x = np.ones((32, 32))
        y = p.forward(x)
        assert y.shape == (32, 32)
        assert np.all(np.isfinite(y))

    def test_center_less_depleted(self):
        """Center pixel (donut=0) should keep full value."""
        p = SaturationDepletion({"depletion_factor": 1.0, "psf_sigma": 2.0})
        x = np.ones((32, 32))
        y = p.forward(x)
        # Center pixel should be close to 1.0 (donut ~ 0 at center)
        assert y[16, 16] > 0.9


class TestBlinkingEmitterModel:
    def test_forward_sparse(self):
        p = BlinkingEmitterModel({"density": 0.5, "photons_per_emitter": 100, "seed": 42})
        x = np.random.rand(16, 16)
        y = p.forward(x)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))

    def test_is_stochastic(self):
        assert BlinkingEmitterModel._is_stochastic is True


class TestEvanescentFieldDecay:
    def test_forward_3d(self):
        p = EvanescentFieldDecay({"penetration_depth": 5.0})
        x = np.ones((8, 16, 16))
        y = p.forward(x)
        assert y.shape == x.shape
        # First slice should have higher value than last
        assert y[0, 0, 0] > y[-1, 0, 0]

    def test_forward_2d_passthrough(self):
        p = EvanescentFieldDecay()
        x = np.ones((16, 16))
        y = p.forward(x)
        np.testing.assert_array_equal(y, x)

    def test_is_linear(self):
        assert EvanescentFieldDecay._is_linear is True


class TestDopplerEstimator:
    def test_forward_3d(self):
        p = DopplerEstimator({"prf": 1000.0, "n_ensembles": 4})
        x = np.random.rand(4, 16, 16)
        y = p.forward(x)
        assert y.shape == (16, 16)

    def test_forward_2d_returns_zeros(self):
        p = DopplerEstimator()
        x = np.ones((16, 16))
        y = p.forward(x)
        assert np.allclose(y, 0.0)


class TestElasticWaveModel:
    def test_forward_scaling(self):
        p = ElasticWaveModel({"wave_speed": 2.0})
        x = np.ones((16, 16))
        y = p.forward(x)
        np.testing.assert_allclose(y, 2.0)

    def test_is_linear(self):
        assert ElasticWaveModel._is_linear is True


class TestDualEnergyBeerLambert:
    def test_forward_shape(self):
        p = DualEnergyBeerLambert()
        x = np.ones((16, 16))
        y = p.forward(x)
        assert y.shape == (2, 16, 16)
        assert np.all(np.isfinite(y))

    def test_forward_values(self):
        p = DualEnergyBeerLambert({"I_0_low": 100, "I_0_high": 200, "mu_low": 0.0, "mu_high": 0.0})
        x = np.zeros((8, 8))
        y = p.forward(x)
        np.testing.assert_allclose(y[0], 100.0)
        np.testing.assert_allclose(y[1], 200.0)


class TestDepthOptics:
    def test_forward_shape(self):
        p = DepthOptics({"focal_length": 50.0, "aperture": 2.8})
        x = np.random.rand(32, 32)
        y = p.forward(x)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))


class TestDiffractionCamera:
    def test_forward_shape(self):
        p = DiffractionCamera()
        x = np.random.rand(32, 32)
        y = p.forward(x)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))
        assert np.all(y >= 0)  # intensity is non-negative

    def test_forward_with_psf(self):
        p = DiffractionCamera({"detector_psf_sigma": 1.0})
        x = np.random.rand(16, 16)
        y = p.forward(x)
        assert y.shape == x.shape


class TestSARBackprojection:
    def test_forward_shape(self):
        p = SARBackprojection({"n_pulses": 8})
        x = np.random.rand(16, 16)
        y = p.forward(x)
        assert y.shape[0] == 8
        assert y.shape[1] == 16
        assert np.all(np.isfinite(y))

    def test_is_linear(self):
        assert SARBackprojection._is_linear is True


class TestParticleAttenuation:
    def test_forward_values(self):
        p = ParticleAttenuation({"I_0": 1000.0, "cross_section": 0.0})
        x = np.ones((8, 8))
        y = p.forward(x)
        np.testing.assert_allclose(y, 1000.0)

    def test_forward_nonzero(self):
        p = ParticleAttenuation()
        x = np.ones((8, 8))
        y = p.forward(x)
        assert np.all(y > 0)


class TestMultipleScatteringKernel:
    def test_forward_shape(self):
        p = MultipleScatteringKernel({"sigma": 1.0})
        x = np.random.rand(16, 16)
        y = p.forward(x)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))

    def test_adjoint_shape(self):
        p = MultipleScatteringKernel({"sigma": 1.0})
        y = np.random.rand(16, 16)
        x_adj = p.adjoint(y)
        assert x_adj.shape == y.shape


class TestVesselFlowContrast:
    def test_forward_3d(self):
        p = VesselFlowContrast({"n_frames": 4})
        x = np.random.rand(4, 16, 16)
        y = p.forward(x)
        assert y.shape == (16, 16)
        assert np.all(np.isfinite(y))

    def test_forward_2d_passthrough(self):
        p = VesselFlowContrast()
        x = np.ones((16, 16))
        y = p.forward(x)
        np.testing.assert_array_equal(y, x)


class TestSpecularReflectionModel:
    def test_forward_adds_specular(self):
        p = SpecularReflectionModel({"specular_strength": 0.1, "roughness": 0.5})
        x = np.ones((8, 8))
        y = p.forward(x)
        # specular adds energy
        assert np.all(y >= x)

    def test_forward_shape(self):
        p = SpecularReflectionModel()
        x = np.random.rand(16, 16)
        y = p.forward(x)
        assert y.shape == x.shape


class TestStructuredLightProjector:
    def test_forward_shape(self):
        p = StructuredLightProjector({"fringe_freq": 4.0})
        x = np.ones((16, 16))
        y = p.forward(x)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))

    def test_is_linear(self):
        assert StructuredLightProjector._is_linear is True


class TestSequenceBlock:
    def test_forward_scaling(self):
        p = SequenceBlock({"TE_ms": 0.0, "TR_ms": 2000.0})
        x = np.ones((8, 8))
        y = p.forward(x)
        # With TE=0, weight = exp(0) = 1.0
        np.testing.assert_allclose(y, 1.0)

    def test_is_linear(self):
        assert SequenceBlock._is_linear is True


class TestPhysiologyDrift:
    def test_forward_3d_adds_drift(self):
        p = PhysiologyDrift({"drift_amplitude": 0.1, "n_components": 3})
        x = np.ones((8, 16, 16))
        y = p.forward(x)
        assert y.shape == x.shape
        # Drift should make y differ from x
        assert not np.allclose(y, x)

    def test_forward_2d_passthrough(self):
        p = PhysiologyDrift()
        x = np.ones((16, 16))
        y = p.forward(x)
        np.testing.assert_array_equal(y, x)


class TestReciprocalSpaceGeometry:
    def test_forward_shape(self):
        p = ReciprocalSpaceGeometry({"sample_tilt_deg": 70.0})
        x = np.random.rand(32, 32)
        y = p.forward(x)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))

    def test_is_nonlinear(self):
        assert ReciprocalSpaceGeometry._is_linear is False


# =========================================================================
# Task 2: Sensor primitive forward tests
# =========================================================================


class TestSPADToFSensor:
    def test_forward_qe(self):
        p = SPADToFSensor({"qe": 0.5})
        x = np.ones((16, 16))
        y = p.forward(x)
        np.testing.assert_allclose(y, 0.5)

    def test_adjoint(self):
        p = SPADToFSensor({"qe": 0.3})
        y = np.ones((16, 16))
        x_adj = p.adjoint(y)
        np.testing.assert_allclose(x_adj, 0.3)

    def test_carrier_type(self):
        assert SPADToFSensor._carrier_type == "photon"


class TestEnergyResolvingDetector:
    def test_forward_qe(self):
        p = EnergyResolvingDetector({"qe": 0.8})
        x = np.ones((16, 16))
        y = p.forward(x)
        np.testing.assert_allclose(y, 0.8)

    def test_carrier_type(self):
        assert EnergyResolvingDetector._carrier_type == "electron"


class TestFiberBundleSensor:
    def test_forward_shape(self):
        p = FiberBundleSensor({"n_cores": 100, "coupling_efficiency": 0.5, "seed": 0})
        x = np.ones((16, 16))
        y = p.forward(x)
        assert len(y) <= 100
        assert np.all(np.isfinite(y))

    def test_coupling_efficiency(self):
        p = FiberBundleSensor({"n_cores": 50, "coupling_efficiency": 1.0, "seed": 0})
        x = np.ones((16, 16))
        y = p.forward(x)
        np.testing.assert_allclose(y, 1.0)

    def test_is_nonlinear(self):
        assert FiberBundleSensor._is_linear is False


class TestTrackDetectorSensor:
    def test_forward_efficiency(self):
        p = TrackDetectorSensor({"efficiency": 0.95})
        x = np.ones((8, 8))
        y = p.forward(x)
        np.testing.assert_allclose(y, 0.95)

    def test_adjoint(self):
        p = TrackDetectorSensor({"efficiency": 0.5})
        y = np.ones((8, 8))
        x_adj = p.adjoint(y)
        np.testing.assert_allclose(x_adj, 0.5)

    def test_carrier_type(self):
        assert TrackDetectorSensor._carrier_type == "abstract"
