"""The problem family splits by patient, and the external validator agrees with the runtime's own judge (two
independent implementations of the same metrics), on real problems."""
import json
from pathlib import Path

import numpy as np
import pytest

from pwm_mode import ldct_family as fam
from pwm_mode import validator_ldct as val

pytestmark = pytest.mark.skipif(not (fam.DATA / "volumes.npz").exists(), reason="LDCT corpus not on this machine")


def test_the_split_is_by_patient_and_disjoint():
    pats, _ = fam.corpus()
    sp = fam.split(pats)
    assert set(sp["development"]).isdisjoint(sp["held_out"])
    assert sorted(sp["development"] + sp["held_out"]) == sorted(pats)
    assert fam.split(list(reversed(pats))) == sp                      # not by position


def test_the_public_workspace_never_holds_the_key(tmp_path):
    pats, _ = fam.corpus()
    spec = fam.make(tmp_path, pats[0], 1, 150.0, 5)
    pub = {p.name for p in (tmp_path / "public" / spec["problem_id"]).rglob("*") if p.is_file()}
    assert pub == {"low_dose.npy", "problem.json"}
    assert set(spec["conditions"]) == {"anatomy", "lesion_hu", "lesion_r"}


def test_the_validator_agrees_with_the_runtime_judge(tmp_path):
    from ai4science.harness.agents.research_agents.runners import domains
    pats, _ = fam.corpus()
    spec = fam.make(tmp_path, pats[0], 1, 150.0, 5)
    key = tmp_path / "keys" / spec["problem_id"]
    low = np.load(key / "low_dose.npy")
    from scipy.ndimage import gaussian_filter
    for rec in (low, gaussian_filter(low, 1.0), gaussian_filter(low, 4.0)):        # untouched, gentle, heavy blur
        seed_ws, run_ws = tmp_path / "seed", tmp_path / "run"
        for d in (seed_ws / "data", run_ws / "data", run_ws / "results"):
            d.mkdir(parents=True, exist_ok=True)
        for n in ("full_dose", "lesion_mask", "lesion_amplitude"):
            np.save(seed_ws / "data" / f"{n}.npy", np.load(key / f"{n}.npy"))
        np.save(run_ws / "data" / "low_dose.npy", low)
        np.save(run_ws / "results" / "reconstruction.npy", rec.astype(np.float32))
        theirs = domains._score_ldct(seed_ws, run_ws)
        ours = val.metrics(key, rec)
        for k in ("psnr", "psnr_before", "lesion_cnr", "lesion_contrast_retained"):
            assert ours[k] == pytest.approx(theirs[k], rel=1e-3, abs=1e-3), k
        assert val.verdict(ours)["passed"] == domains._judge_ldct(theirs).passed


def test_qualification_keeps_noisy_pairs_and_names_the_reason_for_the_rest(monkeypatch):
    monkeypatch.setattr(fam, "noise_ratio", lambda p: {"C1": 2.4, "L1": 1.1}[p])
    q = fam.qualify(["C1", "L1"])
    assert q["qualified"] == ["C1"] and "1.10 < 1.5" in q["excluded"]["L1"]
