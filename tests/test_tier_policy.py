"""Tests for TierPolicy, FourierRelay, and MaxwellInterface."""

import numpy as np
import pytest

from pwm_core.graph.ir_types import PhysicsTier
from pwm_core.graph.tier_policy import TierBudget, TierPolicy


class TestTierPolicy:
    def test_low_budget_gives_tier0(self):
        policy = TierPolicy()
        tier = policy.select_tier("widefield", TierBudget(max_seconds=0.05, accuracy="low"))
        assert tier == PhysicsTier.tier0_geometry

    def test_medium_budget_gives_tier1(self):
        policy = TierPolicy()
        tier = policy.select_tier("widefield", TierBudget(max_seconds=0.5, accuracy="medium"))
        assert tier == PhysicsTier.tier1_approx

    def test_high_budget_gives_tier2(self):
        policy = TierPolicy()
        tier = policy.select_tier("widefield", TierBudget(max_seconds=30.0, accuracy="high"))
        assert tier == PhysicsTier.tier2_full

    def test_maximum_gives_tier3(self):
        policy = TierPolicy()
        tier = policy.select_tier("widefield", TierBudget(max_seconds=1000.0, accuracy="maximum"))
        assert tier == PhysicsTier.tier3_learned

    def test_ct_override_minimum_tier1(self):
        policy = TierPolicy()
        tier = policy.select_tier("ct", TierBudget(max_seconds=0.01, accuracy="low"))
        assert tier == PhysicsTier.tier1_approx

    def test_nerf_override_tier3(self):
        policy = TierPolicy()
        tier = policy.select_tier("nerf", TierBudget(max_seconds=100.0, accuracy="medium"))
        assert tier == PhysicsTier.tier3_learned

    def test_suggest_primitives(self):
        policy = TierPolicy()
        prims = policy.suggest_primitives("widefield", PhysicsTier.tier1_approx)
        assert "conv2d" in prims or "fourier_relay" in prims


class TestFourierRelay:
    def test_forward_shape(self):
        from pwm_core.graph.primitives import FourierRelay
        prim = FourierRelay(params={
            "transfer_function": "free_space",
            "wavelength_m": 0.5e-6,
            "propagation_distance_m": 1e-3,
        })
        x = np.random.rand(32, 32)
        y = prim.forward(x)
        assert y.shape == (32, 32)

    def test_adjoint_dot_product(self):
        """Verify <Ax, y> ≈ <x, A^T y> for the Fourier relay."""
        from pwm_core.graph.primitives import FourierRelay
        prim = FourierRelay(params={
            "transfer_function": "low_pass",
            "cutoff_freq": 1e5,
            "pixel_size_m": 1e-6,
        })
        rng = np.random.default_rng(42)
        x = rng.standard_normal((32, 32))
        y = rng.standard_normal((32, 32))

        Ax = prim.forward(x)
        ATy = prim.adjoint(y)

        inner_Ax_y = float(np.sum(Ax * y))
        inner_x_ATy = float(np.sum(x * ATy))

        assert abs(inner_Ax_y - inner_x_ATy) / max(abs(inner_Ax_y), 1e-10) < 1e-6

    def test_low_pass_removes_high_freq(self):
        from pwm_core.graph.primitives import FourierRelay
        prim = FourierRelay(params={
            "transfer_function": "low_pass",
            "cutoff_freq": 1e4,
            "pixel_size_m": 1e-6,
        })
        x = np.random.rand(32, 32)
        y = prim.forward(x)
        # Output should be smoother (lower energy in high frequencies)
        assert np.std(y) <= np.std(x) + 0.01


class TestMaxwellInterface:
    def test_raises_not_implemented(self):
        from pwm_core.graph.primitives import MaxwellInterface
        prim = MaxwellInterface(params={"backend": "meep"})
        with pytest.raises(NotImplementedError, match="Maxwell solver"):
            prim.forward(np.zeros((32, 32)))

    def test_error_message_includes_backend(self):
        from pwm_core.graph.primitives import MaxwellInterface
        prim = MaxwellInterface(params={"backend": "tidy3d"})
        with pytest.raises(NotImplementedError, match="tidy3d"):
            prim.forward(np.zeros((32, 32)))


class TestPhysicsTierOnPrimitives:
    def test_all_primitives_have_tier(self):
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        for pid, cls in PRIMITIVE_REGISTRY.items():
            tier = getattr(cls, '_physics_tier', None)
            assert tier is not None, f"Primitive '{pid}' has no _physics_tier"

    def test_identity_is_tier0(self):
        from pwm_core.graph.primitives import Identity
        assert Identity._physics_tier == "tier0_geometry"

    def test_conv2d_is_tier1(self):
        from pwm_core.graph.primitives import Conv2d
        assert Conv2d._physics_tier == "tier1_approx"


class TestTierInCompiledGraph:
    def test_compiler_populates_tier(self):
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.graph.graph_spec import OperatorGraphSpec

        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_tier_compile",
            "nodes": [
                {"node_id": "blur", "primitive_id": "conv2d",
                 "params": {"sigma": 2.0, "mode": "reflect"}},
            ],
            "edges": [],
        })
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        tags = graph.node_tags["blur"]
        assert tags.physics_tier is not None


class TestTierSwitch:
    def test_replace_conv2d_with_fourier_relay(self):
        """Substituting conv2d with fourier_relay should still compile."""
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.graph.graph_spec import OperatorGraphSpec

        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_tier_switch",
            "nodes": [
                {"node_id": "relay", "primitive_id": "fourier_relay",
                 "params": {"transfer_function": "low_pass", "cutoff_freq": 1e5,
                           "pixel_size_m": 1e-6}},
            ],
            "edges": [],
        })
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        x = np.random.rand(32, 32)
        y = graph.forward(x)
        assert y.shape == (32, 32)
