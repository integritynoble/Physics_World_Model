"""test_scientific_validity.py — S1-S4 design-time verification tests."""
import pytest
from pwm_core.targeting.scientific_validity import (
    verify_s1_finite_specifiability,
    verify_s2_hadamard_stability,
    verify_s3_approximability,
    verify_s4_certifiability,
    verify_all_s_conditions,
)


class TestS1:
    def test_pass(self):
        ok, msg = verify_s1_finite_specifiability(
            {"omega": {}, "equations": {}, "observables": {}, "tolerance": 0.01}
        )
        assert ok

    def test_fail_missing(self):
        ok, msg = verify_s1_finite_specifiability({"omega": {}})
        assert not ok

    def test_fail_non_numeric_tolerance(self):
        ok, msg = verify_s1_finite_specifiability(
            {"omega": {}, "equations": {}, "observables": {}, "tolerance": "bad"}
        )
        assert not ok
        assert "numeric" in msg


class TestS2:
    def test_pass(self):
        ok, msg = verify_s2_hadamard_stability(
            {"noise_model": "gaussian"},
            forward_model_condition_number=100.0,
        )
        assert ok

    def test_fail_ill_conditioned(self):
        ok, msg = verify_s2_hadamard_stability(
            {"noise_model": "gaussian"},
            forward_model_condition_number=float("inf"),
        )
        assert not ok

    def test_fail_severely_ill_conditioned(self):
        ok, msg = verify_s2_hadamard_stability(
            {"noise_model": "gaussian"},
            forward_model_condition_number=1e15,
        )
        assert not ok
        assert "ill-conditioned" in msg

    def test_fail_no_noise_model(self):
        ok, msg = verify_s2_hadamard_stability({})
        assert not ok
        assert "noise model" in msg

    def test_fail_high_reproducibility_variance(self):
        ok, msg = verify_s2_hadamard_stability(
            {"noise_model": "gaussian"},
            reproducibility_variance=0.05,
        )
        assert not ok
        assert "variance" in msg.lower()

    def test_noise_model_in_extra(self):
        ok, msg = verify_s2_hadamard_stability(
            {"extra": {"noise_model": "poisson"}},
        )
        assert ok


class TestS3:
    def test_pass(self):
        ok, msg = verify_s3_approximability(
            {"primitive_chain": ["Project", "Detect"]}
        )
        assert ok

    def test_fail_no_primitives(self):
        ok, msg = verify_s3_approximability({})
        assert not ok

    def test_fail_non_finite_discretization(self):
        ok, msg = verify_s3_approximability(
            {"primitive_chain": ["Project"]},
            discretization_error=float("inf"),
        )
        assert not ok

    def test_fail_high_discretization_error(self):
        ok, msg = verify_s3_approximability(
            {"primitive_chain": ["Project"]},
            discretization_error=2.0,
        )
        assert not ok

    def test_fail_non_positive_convergence(self):
        ok, msg = verify_s3_approximability(
            {"primitive_chain": ["Project"]},
            mesh_convergence_rate=-0.5,
        )
        assert not ok

    def test_pass_with_primitives_key(self):
        ok, msg = verify_s3_approximability(
            {"primitives": ["Radiate", "Propagate"]}
        )
        assert ok


class TestS4:
    def test_pass(self):
        ok, msg = verify_s4_certifiability(
            {"psnr_db": 30.0, "ssim": 0.9, "runtime_s": 10.0}
        )
        assert ok

    def test_fail_no_metrics(self):
        ok, msg = verify_s4_certifiability({})
        assert not ok

    def test_fail_error_exceeds_tolerance(self):
        ok, msg = verify_s4_certifiability(
            {"psnr_db": 30.0, "runtime_s": 5.0},
            error_bound=0.1,
            tolerance=0.05,
        )
        assert not ok
        assert "exceeds tolerance" in msg

    def test_fail_no_runtime(self):
        ok, msg = verify_s4_certifiability(
            {"psnr_db": 30.0, "ssim": 0.9}
        )
        assert not ok
        assert "runtime_s" in msg


class TestAllConditions:
    def test_all_pass(self):
        results = verify_all_s_conditions(
            corespec={
                "omega": {},
                "equations": {},
                "observables": {},
                "tolerance": 0.01,
            },
            domain_profile={
                "noise_model": "gaussian",
                "primitive_chain": ["Project"],
            },
            runbundle_metrics={"psnr_db": 30.0, "runtime_s": 5.0},
        )
        assert all(v[0] for v in results.values())

    def test_s4_deferred_without_metrics(self):
        results = verify_all_s_conditions(
            corespec={
                "omega": {},
                "equations": {},
                "observables": {},
                "tolerance": 0.01,
            },
            domain_profile={
                "noise_model": "gaussian",
                "primitive_chain": ["Project"],
            },
        )
        assert results["s4"][0] is True
        assert "deferred" in results["s4"][1].lower()

    def test_partial_failure(self):
        results = verify_all_s_conditions(
            corespec={"omega": {}},  # missing fields -> S1 fail
            domain_profile={
                "noise_model": "gaussian",
                "primitive_chain": ["Project"],
            },
        )
        assert not results["s1"][0]
        assert results["s2"][0]
        assert results["s3"][0]
