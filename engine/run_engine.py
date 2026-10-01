"""Does the knowledge engine work? A scripted run on real low-dose CT (no language model yet).

    python3 -m engine.run_engine --out /path/to/run [--workers 8]

What it does, in order (every step written to ``--out``):

1. **Problems.** The low-dose CT family (patient x slice x lesion contrast x lesion radius) from the real paired
   corpus, split by patient into development and held-out (``ldct_family.split``).
2. **L0.** The library starts with the field's principle, twin and benchmark, and no solutions.
3. **Common mode** solves every held-out problem with what the solver ships with: the runtime's adopted defaults
   (owner-signed 2026-08-05). It never reads or writes the library.
4. **Growth.** For one group of conditions at a time, a development search tunes the solver on development patients
   (the validator may be used there: that is what development problems are for), submits the best setting, and the
   validator checks it on a development patient the search did not use. Only then is it accepted: L1, L2, L3.
5. **PWM mode** solves every held-out problem at each library state, starting from the library's best matching
   solution (``Library.best_solution``), or from the defaults when the library has none for those conditions.
6. **The result:** held-out pass rate and lesion CNR for common mode and for PWM mode at L0..L3. The engine works if
   PWM mode's gap over common mode appears once matching solutions exist and grows as the library grows. If it does
   not, that is the finding.

Budget is matched per held-out problem: one solver run and one validation, in both arms. The search cost is paid
once, on development problems, and reported separately.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pwm_mode import ldct_family as fam          # noqa: E402
from pwm_mode import validator_ldct as val       # noqa: E402
from pwm_mode.library import Library             # noqa: E402

_SOLVER = Path(fam.RUNTIME_SRC) / "ai4science/harness/agents/research_agents/runners/payload/ldct/run_solver.py"
DEFAULTS = {"sigma_s": 3.0, "sigma_r_scale": 0.3, "iters": 3, "radius": 4}     # the adopted defaults
GRID = {"sigma_s": [1.5, 3.0, 4.5, 6.0], "sigma_r_scale": [0.3, 0.6, 0.9, 1.4], "iters": [1, 2, 3, 5],
        "radius": [2, 3, 4, 5]}
STEPS = [(150.0,), (100.0,), (200.0,)]          # each growth step adds the solutions for one lesion contrast


def _solver():
    spec = importlib.util.spec_from_file_location("ldct_run_solver", _SOLVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solve(public_dir: str, params: dict) -> np.ndarray:
    low = np.load(Path(public_dir) / "data" / "low_dose.npy")
    s = _solver()
    return s.bilateral(low, sigma_s=float(params["sigma_s"]), guide_scale=float(params["sigma_r_scale"]),
                       iters=int(round(params["iters"])), radius=int(round(params["radius"])))


def run_one(args):
    store, pid, params = args
    rec = solve(str(Path(store) / "public" / pid), params)
    v = val.validate(Path(store) / "keys" / pid, rec)
    return {"problem": pid, "params": params, "passed": v["passed"], "objective": v["objective"],
            "metrics": v["metrics"]}


def seed_library(lib: Library):
    return lib.seed([
        {"id": "ldct-principle-edge-preserving", "layer": "L1", "field": "ldct",
         "title": "Edge-preserving restoration for low-dose CT",
         "summary": "noise falls, a low-contrast edge does not: guided bilateral weights on a pre-smoothed guide",
         "conditions": {}, "payload": {}},
        {"id": "ldct-twin-paired-tcia", "layer": "L2", "field": "ldct",
         "title": "Paired full-dose / low-dose reconstructions (TCIA LDCT-and-Projection-data)",
         "summary": "the same acquisition reconstructed at full and reduced dose; no simulated noise",
         "conditions": {}, "payload": {"source": "TCIA LDCT-and-Projection-data"}},
        {"id": "ldct-benchmark-lesion", "layer": "L3", "field": "ldct",
         "title": "Restore without losing an inserted low-contrast lesion",
         "summary": "PSNR above the low-dose slice; lesion CNR >= 3; half the contrast retained",
         "conditions": {}, "payload": {"validator": val.NAME}},
    ], why="L0: principle, twin and benchmark; no solutions")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--candidates", type=int, default=24, help="settings tried per condition in development")
    a = ap.parse_args(argv)
    out = Path(a.out)
    store, lib = out / "problems", Library(out / "library")
    log = open(out / "engine.log", "a") if out.exists() else None
    out.mkdir(parents=True, exist_ok=True)
    log = log or open(out / "engine.log", "a")
    say = lambda *x: (print(*x, flush=True), print(*x, file=log, flush=True))
    t0 = time.time()

    pats, _ = fam.corpus()
    sp = fam.split(pats)
    problems = fam.build_family(store)
    by_pat = {}
    for p in problems:
        by_pat.setdefault(p["patient"], []).append(p)
    dev = [p for q in sp["development"] for p in by_pat.get(q, [])]
    test = [p for q in sp["held_out"] for p in by_pat.get(q, [])]
    (out / "split.json").write_text(json.dumps(sp, indent=1))
    say(f"problems: {len(problems)} · development patients {sp['development']} ({len(dev)}) · "
        f"held-out patients {sp['held_out']} ({len(test)})")
    st0 = seed_library(lib)
    say(f"L0 digest {st0['digest'][:16]}")

    pool = ProcessPoolExecutor(max_workers=a.workers)
    common = list(pool.map(run_one, [(str(store), p["problem_id"], DEFAULTS) for p in test]))
    results = {"common": common, "pwm": {}, "growth": []}

    def pwm_at(state_n):
        jobs = []
        for p in test:
            best = lib.best_solution("ldct", p["conditions"], state=state_n)
            match = best and all(best["conditions"].get(k) == v for k, v in p["conditions"].items()
                                 if k in ("lesion_hu", "lesion_r"))
            jobs.append((str(store), p["problem_id"], best["payload"]["params"] if match else DEFAULTS))
        return list(pool.map(run_one, jobs))

    results["pwm"]["L0"] = pwm_at(st0["n"])
    rng = random.Random(7)
    space = list(itertools.product(*GRID.values()))
    dev_pats = sp["development"]
    search_pat, check_pat = dev_pats[0], dev_pats[-1]
    for step, contrasts in enumerate(STEPS, 1):
        for hu in contrasts:
            for r in fam.RADII:
                cond = fam.conditions(search_pat, hu, r)
                cands = [dict(zip(GRID, c)) for c in rng.sample(space, min(a.candidates, len(space)))]
                cands.append(dict(DEFAULTS))
                search = [p for p in by_pat[search_pat] if p["conditions"] == cond]
                runs = list(pool.map(run_one, [(str(store), p["problem_id"], c) for c in cands for p in search]))
                score = {}
                for rr in runs:
                    k = json.dumps(rr["params"], sort_keys=True)
                    score.setdefault(k, []).append(rr)
                ranked = sorted(score.items(), key=lambda kv: (-sum(x["passed"] for x in kv[1]),
                                                              -float(np.mean([x["objective"] for x in kv[1]]))))
                best_params = json.loads(ranked[0][0])
                sub = lib.submit(field="ldct", layer="L4", id=f"ldct-sol-c{int(hu)}-r{r}",
                                 title=f"Guided bilateral tuned for a {int(hu)} HU, {r} px lesion",
                                 summary="setting found by development search; validated on another patient",
                                 payload={"method": "guided-bilateral (runtime run_solver)", "params": best_params},
                                 conditions=cond, source_problems=[p["problem_id"] for p in search],
                                 submitter="engine-search")
                checks = [run_one((str(store), p["problem_id"], best_params)) for p in by_pat[check_pat]
                          if p["conditions"]["lesion_hu"] == hu and p["conditions"]["lesion_r"] == r]
                passed = all(c["passed"] for c in checks) and bool(checks)
                validation = {"passed": passed, "validator": val.NAME,
                              "metrics": {"mean_cnr": float(np.mean([c["metrics"]["lesion_cnr"] for c in checks]))},
                              "objective": float(np.mean([c["objective"] for c in checks])),
                              "problems": [c["problem"] for c in checks]}
                if passed:
                    st = lib.accept(sub["id"], validation)
                    say(f"step {step}: accepted {sub['artifact']['id']} {best_params} → L{st['n']}")
                else:
                    lib.reject(sub["id"], "the validator did not pass it on the check patient")
                    say(f"step {step}: rejected {sub['artifact']['id']} (validator)")
                results["growth"].append({"step": step, "condition": cond, "params": best_params,
                                          "accepted": passed, "search_runs": len(runs), "validation": validation})
        st = lib.state()
        results["pwm"][f"after step {step} (L{st['n']})"] = pwm_at(st["n"])

    def summary(rows):
        return {"n": len(rows), "pass_rate": float(np.mean([r["passed"] for r in rows])),
                "mean_cnr": float(np.mean([r["metrics"]["lesion_cnr"] for r in rows])),
                "mean_retained": float(np.mean([r["metrics"]["lesion_contrast_retained"] for r in rows]))}
    report = {"split": sp, "common": summary(common), "pwm": {k: summary(v) for k, v in results["pwm"].items()},
              "library_states": [lib.state(n) for n in range(lib.head() + 1)], "seconds": round(time.time() - t0)}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    say(json.dumps({"common": report["common"], "pwm": report["pwm"]}, indent=1))
    pool.shutdown()


if __name__ == "__main__":
    main()
