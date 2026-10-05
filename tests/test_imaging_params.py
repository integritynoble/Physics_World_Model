"""LIB-40 IMG-10 part 2: template parameters act (needs the pwm_core with results/IMG-10/pwm_core_params.patch applied).

The tests skip when PWM_CORE_SRC is the unpatched 3899e328: that tree still has dead parameters by construction.
"""
import inspect

import pytest

from pwm_mode.imaging import core, sensitivity

pytest.importorskip("yaml")
if not core.PWM_CORE_SRC.exists():
    pytest.skip("pwm_core not available", allow_module_level=True)

core._ensure_path()
from pwm_core.graph.primitives import PRIMITIVE_REGISTRY  # noqa: E402

if "detector_offset" not in inspect.getsource(PRIMITIVE_REGISTRY["ct_radon"]):
    pytest.skip("pwm_core without the IMG-10 parameter patch", allow_module_level=True)

CHANGED = ["ct_graph_v2", "cbct_graph_v2", "ultrasound_graph_v2", "photoacoustic_graph_v2", "cassi_graph_v2",
           "cacti_graph_v2", "elastography_graph_v2", "compressive_holography_graph_v2", "sem_graph_v2",
           "tem_graph_v2", "fmri_graph_v2", "fluorescence_microscopy_graph_v2", "acoustic_microscopy_graph_v1"]
# (template, node, parameter) that must act, and where
MUST_ACT = [("ct_graph_v2", "radon", "detector_offset"), ("ct_graph_v2", "radon", "angle_offset"),
            ("cbct_graph_v2", "radon", "detector_offset"), ("ultrasound_graph_v2", "propagate", "speed_of_sound"),
            ("photoacoustic_graph_v2", "propagate", "speed_of_sound"), ("cassi_graph_v2", "modulate", "mask_dx"),
            ("cassi_graph_v2", "modulate", "mask_dy"), ("cacti_graph_v2", "coded_aperture", "mask_dx"),
            ("elastography_graph_v2", "elastic", "wave_speed"), ("acoustic_microscopy_graph_v1", "propagate", "distance"),
            ("compressive_holography_graph_v2", "propagate", "distance"),
            ("fluorescence_microscopy_graph_v2", "fluorophore", "yield_coeff"),
            ("fmri_graph_v2", "sequence", "flip_angle_deg"), ("tem_graph_v2", "source", "beam_current_na")]


@pytest.fixture(scope="module")
def report():
    return {t["template"]: t for t in sensitivity.scan_templates(CHANGED)["templates"]}


def test_changed_templates_run(report):
    assert all(report[t]["runs"] for t in CHANGED), [t for t in CHANGED if not report[t]["runs"]]


def test_no_dead_parameter_in_changed_templates(report):
    dead = [(t, r["node"], r["param"]) for t in CHANGED for r in report[t].get("params", []) if r["status"] == "dead"]
    assert not dead, dead


@pytest.mark.parametrize("tid,node,param", MUST_ACT)
def test_new_parameter_acts(report, tid, node, param):
    rows = [r for r in report[tid]["params"] if r["node"] == node and r["param"] == param]
    assert rows, (tid, node, param)
    assert rows[0]["status"] == "acts", rows[0]


def test_electron_templates_depend_on_the_object(report):
    for tid in ("sem_graph_v2", "tem_graph_v2"):
        assert report[tid]["output"]["ignores_object"] is False, tid


def test_removed_parameters_are_recorded():
    T = core.contrib("graph_templates.yaml")["templates"]
    assert T["tem_graph_v2"]["metadata"]["unmodelled_params"]["source.accelerating_voltage_kv"] == 200.0
    assert "accelerating_voltage_kv" not in next(n for n in T["tem_graph_v2"]["nodes"] if n["node_id"] == "source")["params"]
