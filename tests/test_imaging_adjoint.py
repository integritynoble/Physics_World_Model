"""C5 adjoint (dot-product) tests for every running graph template of the imaging library (LIB-40 item IMG-07).

For each of the 192 templates that compile, the forward plan is cut into maximal runs of consecutive linear primitives and
each run is tested as a linear operator: <Ax, y> = <x, A^T y> on seeded standard-normal x and y (3 draws, seed
adjoint.DEFAULT_SEED), at the template's own x_shape, with relative error below adjoint.TOLERANCE (1e-8). Exact float64
adjoints give <= 3e-14 here and every failing one gives >= 0.1, so the verdicts do not depend on the tolerance.

156 templates pass. 36 fail and 5 do not compile; they are listed below with the primitive at fault, as strict xfails:
CI stays green while the library is as recorded, and goes red when a fix lands (XPASS) or when a template fails in a
different place, so the table cannot go stale. Cause per primitive: engine/results/IMG-07-adjoint.md.

Needs the full pwm_core (PWM_CORE_SRC); skipped without it. Runs in about 10 s after a ~20 s import.
"""
from functools import lru_cache

import numpy as np
import pytest

pytest.importorskip("yaml")
from pwm_mode.imaging import adjoint, core  # noqa: E402

if not core.PWM_CORE_SRC.exists():
    pytest.skip("pwm_core not available", allow_module_level=True)

pytestmark = pytest.mark.filterwarnings("ignore:Casting complex values")

# template -> primitives whose own dot test fails (found by the per-node diagnosis of the failing run)
KNOWN_FAILURES = {
    # ct_radon: rotation-based "approximate" adjoint above 32x32 (exact matrix only for H*W <= 1024)
    "ct_graph_v1": ("ct_radon",), "ct_graph_v2": ("ct_radon",), "cbct_graph_v2": ("ct_radon",),
    "brachytherapy_img_graph_v1": ("ct_radon",), "cryo_et_graph_v1": ("ct_radon",),
    "digital_breast_tomo_graph_v1": ("ct_radon",), "industrial_ct_graph_v1": ("ct_radon",),
    "portal_imaging_graph_v1": ("ct_radon",), "proton_therapy_img_graph_v1": ("ct_radon",),
    "xray_ndt_graph_v1": ("ct_radon",),
    # electron_beam_source: ignores its input (current * ones_like(x)), so it is affine, not linear
    "sem_graph_v2": ("electron_beam_source",), "tem_graph_v2": ("electron_beam_source",),
    "electron_tomography_graph_v2": ("electron_beam_source",), "stem_graph_v2": ("electron_beam_source",),
    "cryo_em_graph_v2": ("electron_beam_source",),
    # mri_kspace: fft2 forward, ifft2 adjoint; ifft2 carries 1/(H*W), so the adjoint is 1/4096 of the true one
    "mri_graph_v1": ("mri_kspace",), "mri_graph_v2": ("mri_kspace",), "diffusion_mri_graph_v2": ("mri_kspace",),
    # ... and sequence_block casts the complex k-space to float64 (drops the imaginary part)
    "fmri_graph_v2": ("mri_kspace", "sequence_block"), "mrs_graph_v2": ("mri_kspace", "sequence_block"),
    # acoustic_propagation / emission_projection / sar_backprojection: forward rotates and sums, "adjoint" tiles and
    # rotates back (not the transpose); acoustic_propagation also divides by n_angles
    "photoacoustic_graph_v2": ("acoustic_propagation",), "doppler_ultrasound_graph_v2": ("acoustic_propagation",),
    "elastography_graph_v2": ("acoustic_propagation",), "ultrasound_graph_v2": ("acoustic_propagation", "beamform_delay"),
    "pet_graph_v2": ("emission_projection",), "spect_graph_v2": ("emission_projection",),
    "sar_graph_v2": ("sar_backprojection",),
    # motion_warp: exact for integer shifts, not for fractional ones (0.5 and 0.3 in these two templates)
    "light_field_graph_v1": ("motion_warp",), "integral_graph_v1": ("motion_warp",),
    # photon_sensor: dark_current makes it affine (spc_graph_v2); on complex input it drops the imaginary part
    "spc_graph_v2": ("photon_sensor",),
    # interference: adjoint raises NotImplementedError although is_linear is True; photon_sensor then drops the imaginary part
    "compressive_holography_graph_v2": ("interference", "photon_sensor"),
    # adjoint output has the wrong shape (ignores axis / uses a fixed T, n_frames or n_elements)
    "panorama_graph_v2": ("frame_integration",), "flim_graph_v2": ("temporal_mask",),
    "fluoroscopy_graph_v2": ("fluoro_temporal_integrator",), "angiography_graph_v2": ("fluoro_temporal_integrator",),
    "sonar_graph_v2": ("beamform_delay",),
}
# The five sampled-Fourier templates no longer fail to compile: IMG-01 rewrote them to mri_kspace (merged into LIB-40),
# so they now share the MRI template's known mri_kspace adjoint failure.
SAMPLED_FOURIER = ("eht_imaging_graph_v1", "insar_graph_v1", "radio_astronomy_graph_v1", "radio_interferometry_graph_v1",
                   "xray_crystallography_graph_v1")
KNOWN_FAILURES.update({n: ("mri_kspace",) for n in SAMPLED_FOURIER})
COMPILE_ERRORS = ()

TEMPLATES = adjoint.load_templates()


@lru_cache(maxsize=None)
def result(name):
    return adjoint.test_template(TEMPLATES[name])


def culprits(res):
    return tuple(sorted({n["primitive"] for s in res.get("segments", []) for n in s.get("per_node", []) if n["status"] != "pass"}))


def _param(name):
    if name in KNOWN_FAILURES:
        return pytest.param(name, marks=pytest.mark.xfail(strict=True, reason="known: " + ", ".join(KNOWN_FAILURES[name])))
    return name


@pytest.mark.parametrize("name", [_param(n) for n in sorted(TEMPLATES) if n not in COMPILE_ERRORS])
def test_dot_product(name):
    r = result(name)
    assert r["status"] == "pass", [(s["primitives"], s.get("reason"), s.get("max_relative_error")) for s in r["segments"] if s["status"] != "pass"]


@pytest.mark.parametrize("name", sorted(KNOWN_FAILURES))
def test_known_failure_is_where_recorded(name):
    r = result(name)
    assert r["status"] == "fail", f"{name} now passes; remove it from KNOWN_FAILURES"
    assert culprits(r) == tuple(sorted(KNOWN_FAILURES[name]))


@pytest.mark.parametrize("name", COMPILE_ERRORS)
def test_compile_error_is_the_recorded_one(name):
    r = result(name)
    assert r["status"] == "compile_error" and "subsample" in r["error"], f"{name} now: {r['status']}; update COMPILE_ERRORS"


def test_tables_name_real_templates():
    assert set(KNOWN_FAILURES) | set(COMPILE_ERRORS) <= set(TEMPLATES)
    assert not set(KNOWN_FAILURES) & set(COMPILE_ERRORS)
    assert len(TEMPLATES) == 197


def test_every_linear_primitive_is_tested():
    # a linear primitive that is in no tested run would be silently skipped; the 5 compile errors have no plan
    for name in TEMPLATES:
        r = result(name)
        if r["status"] == "compile_error":
            continue
        tested = sum(len(s["nodes"]) for s in r["segments"])
        assert tested == sum(r["linear_flags"]), name


def test_tolerance_is_not_what_decides():
    passing = [s["max_relative_error"] for n in TEMPLATES for s in result(n).get("segments", []) if s["status"] == "pass"]
    failing = [s["max_relative_error"] for n in TEMPLATES for s in result(n).get("segments", [])
               if s["status"] == "fail" and "max_relative_error" in s]
    assert max(passing) < 1e-12 and min(failing) > 1e-2


def test_result_is_seeded_and_repeatable():
    t = TEMPLATES["widefield_graph_v2"]
    a, b = adjoint.test_template(t), adjoint.test_template(t)
    assert [s["max_relative_error"] for s in a["segments"]] == [s["max_relative_error"] for s in b["segments"]]
    assert adjoint.test_template(t, seed=7)["status"] == "pass"


# ---- the checker has power: it must accept an exact adjoint and reject each kind of wrong one -------------------------

class _P:
    is_linear = True
    primitive_id = "synthetic"

    def __init__(self, fwd, adj):
        self.forward, self.adjoint = fwd, adj


def _dot(fwd, adj, shape=(6, 5)):
    return adjoint.dot_test([("n", _P(fwd, adj))], np.zeros(shape), seed=1)


def test_checker_accepts_an_exact_adjoint_and_rejects_wrong_ones():
    m = np.random.default_rng(0).standard_normal((7, 30))
    f = lambda x: (m @ x.ravel()).reshape(7)  # noqa: E731
    assert _dot(f, lambda y: (m.T @ y).reshape(6, 5))["status"] == "pass"
    assert _dot(f, lambda y: 2 * (m.T @ y).reshape(6, 5))["status"] == "fail"                  # wrong scale
    r = _dot(lambda x: 3 * x, lambda y: 3.001 * y)
    assert r["status"] == "fail" and r["max_relative_error"] > 1e-4                              # small, still caught
    assert _dot(lambda x: x + 1.0, lambda y: y)["status"] == "fail"                              # affine
    r = _dot(lambda x: x.sum(axis=0), lambda y: np.tile(y, (3, 1)))
    assert r["status"] == "fail" and "shape" in r["reason"]                                      # shape contract

    def none(y):
        raise NotImplementedError("no adjoint")
    assert "not implemented" in _dot(lambda x: x, none)["reason"]


def test_checker_handles_complex_data():
    z = np.zeros((4, 4), dtype=complex)
    d = np.random.default_rng(2).standard_normal((4, 4)) + 1j * np.random.default_rng(3).standard_normal((4, 4))
    assert adjoint.dot_test([("n", _P(lambda x: d * x, lambda y: np.conj(d) * y))], z, 1)["status"] == "pass"
    assert adjoint.dot_test([("n", _P(lambda x: d * x, lambda y: d * y))], z, 1)["status"] == "fail"   # forgot the conjugate
    # float cast drops the imaginary part: not complex-linear, but self-adjoint under the real inner product
    r = adjoint.dot_test([("n", _P(lambda x: 2.0 * x.real, lambda y: 2.0 * y.real))], z, 1)
    assert r["status"] == "fail" and r["real_inner_product_error"] < 1e-12


# ---- the explanations in the report, pinned at primitive level ---------------------------------------------------------

def _prim(pid, params):
    return core.registry()[pid](params)


def test_ct_radon_is_exact_with_its_matrix_and_not_without():
    small = adjoint.dot_test([("n", _prim("ct_radon", {"H": 16, "W": 16, "n_angles": 8}))], np.zeros((16, 16)), 1)
    big = adjoint.dot_test([("n", _prim("ct_radon", {"H": 40, "W": 40, "n_angles": 8}))], np.zeros((40, 40)), 1)
    assert small["status"] == "pass" and big["status"] == "fail" and big["max_relative_error"] > 0.01


def test_mri_kspace_adjoint_is_the_true_one_divided_by_hw():
    r = adjoint.dot_test([("n", _prim("mri_kspace", {"H": 16, "W": 16}))], np.zeros((16, 16)), 1)
    assert r["status"] == "fail" and r["rhs_over_lhs"]["constant"]
    assert r["rhs_over_lhs"]["mean"] == pytest.approx(1 / 256)


def test_motion_warp_is_exact_for_integer_shifts_only():
    x0 = np.zeros((16, 16))
    assert adjoint.dot_test([("n", _prim("motion_warp", {"dx": 2.0, "dy": 1.0}))], x0, 1)["status"] == "pass"
    assert adjoint.dot_test([("n", _prim("motion_warp", {"dx": 0.5, "dy": 0.5}))], x0, 1)["status"] == "fail"


def test_electron_beam_source_is_affine():
    lin = adjoint.linearity([("n", _prim("electron_beam_source", {"beam_current_na": 1.0}))], np.zeros((8, 8)), 1)
    assert lin["response_to_zero"] > 0.5 and lin["additivity_error"] > 0.01
