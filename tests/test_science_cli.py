"""The science task's tools: make keeps the key out of the worker's folder; validate exits 0 / 1 / 2."""
import json

import numpy as np
import pytest

from pwm_mode import ldct_family as fam
from pwm_mode import science_cli as cli

pytestmark = pytest.mark.skipif(not (fam.DATA / "volumes.npz").exists(), reason="LDCT corpus not on this machine")


def test_make_then_validate(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PWM_MODE_HOME", str(tmp_path / "home"))
    pats, _ = fam.corpus()
    work = tmp_path / "work"
    assert cli.main(["make", "--patient", pats[0], "--slice", "1", "--hu", "200", "--r", "7", "--into", str(work)]) == 0
    made = json.loads(capsys.readouterr().out)
    assert sorted(p.name for p in work.rglob("*") if p.is_file()) == ["low_dose.npy", "problem.json"]
    low = np.load(work / "data" / "low_dose.npy")
    from scipy.ndimage import gaussian_filter
    np.save(work / "heavy.npy", gaussian_filter(low, 6.0))           # blurs the lesion away: must fail
    assert cli.main(["validate", "--problem", made["problem_id"], "--reconstruction", str(work / "heavy.npy")]) == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False
    np.save(work / "bad.npy", low[:10])                               # wrong shape: nothing to judge
    assert cli.main(["validate", "--problem", made["problem_id"], "--reconstruction", str(work / "bad.npy")]) == 2
    capsys.readouterr()
    assert cli.main(["validate", "--problem", "../etc", "--reconstruction", str(work / "bad.npy")]) == 2
