"""Build and evaluate the synthetic, NIST-isotope-backed APT simulation judge."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import data, judges
from .atom_probe_model import ISOTOPES, MODE_SIGMA_DA, forward_spectrum

ROOT = Path(__file__).resolve().parents[3]
KEY_DIR = Path(os.environ.get("IPWM24_KEYS", "/tmp/ipwm24-keys"))
PUBLIC_DIR = Path(os.environ.get("IPWM24_PUBLIC", "/tmp/ipwm24-public"))
RESULT_DIR = ROOT / "results" / "IPWM-24" / "atom_probe"
CRITERION = {"relative_l2_max": 0.10}
ELEMENTS = ("N", "Al", "Ti", "V", "Cr", "Fe", "Ni")
GRID = np.arange(0.0, 70.0, 0.025, dtype=np.float64)
MODE_SIGMA = dict(MODE_SIGMA_DA)
UNIT_KIND = "synthetic alloy spectrum"


def _salt() -> str:
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    p = KEY_DIR / "SALT"
    if not p.exists():
        p.write_text(secrets.token_hex(32) + "\n")
    return p.read_text().strip()


def _rng(salt: str, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256((salt + "|" + "|".join(parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _unit_id(salt: str, index: int) -> str:
    return hashlib.sha256(f"{salt}|atom_probe|simulate|{index}".encode()).hexdigest()[:16]


def _input_case(index: int, salt: str) -> dict[str, Any]:
    rng = _rng(salt, "public-case", str(index))
    # Composition envelope follows Fe-Cr-Ni stainless alloy with minor Ti/N;
    # additional Al/V content broadens isotope-overlap conditions.
    base = np.asarray([0.015, 0.005, 0.006, 0.003, 0.15, 0.64, 0.181], dtype=float)
    concentration = float(rng.uniform(250.0, 800.0))
    fractions = rng.dirichlet(base * concentration)
    charge2 = {el: float(rng.uniform(0.08, 0.42)) for el in ELEMENTS}
    mode = ("high", "nominal", "low")[index % 3]
    return {
        "atomic_fraction": {el: float(v) for el, v in zip(ELEMENTS, fractions)},
        "charge2_fraction": charge2,
        "detector_mode": mode,
        "ions_evaporated": int(rng.integers(120000, 180001)),
        "detector_efficiency": float(rng.uniform(0.48, 0.68)),
        "background_counts_per_bin": float(rng.uniform(0.35, 0.9)),
        "bin_width_da": 0.025,
        "mass_to_charge_grid": GRID.tolist(),
        "sample_family": "Fe-Cr-Ni alloy with minor Ti/N; synthetic composition",
    }


def _reference(case: dict[str, Any], salt: str, unit: str) -> np.ndarray:
    """Independent reference calculation plus hidden shot noise and instrument residuals."""
    rng = _rng(salt, "reference-key", unit)
    sigma = MODE_SIGMA[case["detector_mode"]] * float(rng.uniform(0.975, 1.025))
    mass_shift = float(rng.uniform(-0.003, 0.003))
    gain = float(rng.uniform(0.995, 1.005))
    step = float(case["bin_width_da"])
    expected = np.zeros_like(GRID)
    detected = float(case["ions_evaporated"]) * float(case["detector_efficiency"])
    for element in ELEMENTS:
        fraction = float(case["atomic_fraction"][element])
        q2 = float(case["charge2_fraction"][element])
        for mass, abundance in ISOTOPES[element]:
            for charge, qprob in ((1, 1.0 - q2), (2, q2)):
                z = (GRID - (float(mass) / charge + mass_shift)) / sigma
                expected += (detected * fraction * float(abundance) * qprob
                             * np.exp(-0.5 * z * z) * step / (sigma * np.sqrt(2.0 * np.pi)))
    expected = np.maximum(expected * gain + float(case["background_counts_per_bin"]), 0.0)
    return rng.poisson(expected).astype(np.float64)


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _rate(verdicts: list[dict[str, Any]]) -> float:
    return float(np.mean([v["pass"] for v in verdicts])) if verdicts else 0.0


def build_and_evaluate() -> tuple[dict[str, Any], dict[str, Any]]:
    salt = _salt()
    devdir = PUBLIC_DIR / "atom_probe" / "simulate" / "dev"
    testdir = PUBLIC_DIR / "atom_probe" / "simulate" / "test"
    keydir = KEY_DIR / "atom_probe" / "simulate"
    for d in (devdir, testdir, keydir):
        d.mkdir(parents=True, exist_ok=True)

    inputs, keys, key_hashes = [], {}, {}
    for i in range(20):
        uid = _unit_id(salt, i)
        public_input = _input_case(i, salt)
        public_input["unit"] = uid
        inputs.append(public_input)
        ref = _reference(public_input, salt, uid)
        if i < 8:
            train_public = {**public_input, "reference_spectrum": ref.tolist()}
            _write(devdir / f"{uid}.json", train_public)
        else:
            _write(testdir / f"{uid}.json", public_input)

    dev_lookup = {r["unit"]: r for r in (json.loads(p.read_text()) for p in devdir.glob("*.json"))}
    dev_rows = [dev_lookup[row["unit"]] for row in inputs[:8]]
    # Rebuilding with the same salt yields the same input/key; clean stale files from a different salt.
    dev_curves = np.asarray([r["reference_spectrum"] for r in dev_rows], dtype=np.float64)
    if dev_curves.shape != (8, len(GRID)):
        raise RuntimeError(f"expected 8 development spectra, got {dev_curves.shape}")
    forbidden = sorted({data.curve_sha256(c) for c in dev_curves})
    for i in range(8, 20):
        uid = inputs[i]["unit"]
        ref = _reference(inputs[i], salt, uid)
        keypath = keydir / f"{uid}.npz"
        np.savez_compressed(keypath, reference_spectrum=ref,
                            forbidden_public_spectrum_sha256=np.asarray(forbidden, dtype="U64"))
        key_hashes[keypath.name] = hashlib.sha256(keypath.read_bytes()).hexdigest()
        keys[uid] = {"reference_spectrum": ref, "forbidden_public_spectrum_sha256": forbidden}

    # Select one global resolution width on development units only.
    width_grid = (0.025, 0.035, 0.050, 0.070, 0.090)
    scores = []
    for sigma in width_grid:
        vs = [judges.judge_atom_probe_simulate({"spectrum": forward_spectrum(row, sigma)},
                                               {"reference_spectrum": dev_curves[j],
                                                "forbidden_public_spectrum_sha256": []}, CRITERION)
              for j, row in enumerate(inputs[:8])]
        scores.append({"setting": {"sigma_da": sigma}, "dev_pass_rate": _rate(vs)})
    best_setting = max(scores, key=lambda x: (x["dev_pass_rate"], -abs(x["setting"]["sigma_da"] - 0.05)))["setting"]

    def score_method(name: str, fn) -> dict[str, Any]:
        rows = []
        for i in range(8, 20):
            case = inputs[i]
            sub = fn(case, i)
            verdict = judges.judge_atom_probe_simulate(sub, keys[case["unit"]], CRITERION)
            rows.append({"unit": case["unit"], **verdict})
        return {"pass_rate": _rate(rows), "n": len(rows), "verdicts": rows}

    mean_train = dev_curves.mean(axis=0)
    blur_train = gaussian_filter1d(mean_train, sigma=2.0)
    constant = np.full_like(GRID, float(mean_train.mean()))
    copy_public = dev_curves[0]
    methods = {
        "trivial:input_unchanged": score_method("input_unchanged", lambda c, i: c),
        "trivial:constant": score_method("constant", lambda c, i: {"spectrum": constant}),
        "trivial:zeros": score_method("zeros", lambda c, i: {"spectrum": np.zeros_like(GRID)}),
        "trivial:blur": score_method("blur", lambda c, i: {"spectrum": blur_train}),
        "trivial:copy_public": score_method("copy_public", lambda c, i: {"spectrum": copy_public}),
        "best_fixed(setting chosen on dev)": score_method("best_fixed", lambda c, i: {
            "spectrum": forward_spectrum(c, best_setting["sigma_da"])}),
        "adaptive_reference_solver": score_method("adaptive", lambda c, i: {"spectrum": forward_spectrum(c)}),
    }
    try:
        from pwm_mode.imaging import lifecycle
        from pwm_mode.imaging.lifecycle import test_object
        y = lifecycle.simulate("atom_probe", test_object("atom_probe", seed=2026))["y"]
        methods["baseline_pwm_core"] = score_method("baseline", lambda c, i: {"spectrum": y})
    except Exception as exc:  # baseline failures are recorded as failures
        methods["baseline_pwm_core"] = {"pass_rate": 0.0, "n": 12,
            "verdicts": [{"unit": inputs[i]["unit"], "pass": False, "metrics": {},
                          "reasons": [f"pwm_core baseline unavailable: {type(exc).__name__}: {exc}"]}
                         for i in range(8, 20)]}

    trivial_rates = {name.removeprefix("trivial:"): v["pass_rate"]
                     for name, v in methods.items() if name.startswith("trivial:")}
    best_trivial, best_trivial_rate = max(trivial_rates.items(), key=lambda x: x[1])
    isotope_file = Path(__file__).with_name("nist_isotopes.json")
    isotope_sha = hashlib.sha256(isotope_file.read_bytes()).hexdigest()
    data_manifest = {
        "reference_data": "NIST Atomic Weights and Isotopic Compositions (N, Al, Ti, V, Cr, Fe, Ni)",
        "url": "https://pml.nist.gov/cgi-bin/Compositions/stand_alone.pl?ascii=ascii",
        "license": "U.S. federal government reference data; public domain",
        "retained_table": {"path": "pwm_mode/imaging/diffraction/nist_isotopes.json", "sha256": isotope_sha},
        "unit_kind": UNIT_KIND,
        "reference_generation": "independent Gaussian convolution of NIST isotope lines with Poisson shot noise; random hidden mass-axis/gain residuals are keyed by secret salt",
    }
    key_manifest = {name: digest for name, digest in sorted(key_hashes.items())}
    result = {
        "methods": methods,
        "setting_scores_dev": scores,
        "best_fixed_setting": best_setting,
        "development": {"n": 8, "unit_kind": UNIT_KIND},
        "held_out": {"n": 12, "unit_kind": UNIT_KIND},
        "limits": [
            "Synthetic monoatomic-ion spectra only; no molecular ions, preferential evaporation, detector pile-up, or local magnification.",
            "The reference is a physics-derived synthetic spectrum, not a measured APT detector event stream.",
            "Isotope overlaps are represented by NIST masses and natural abundances; alloy-specific isotope fractionation is not modeled.",
        ],
    }
    version = {
        "task": "atom_probe/simulate",
        "criterion_version": judges.CRITERION_VERSION,
        "criterion": CRITERION,
        "judge_type": "atom_probe_mass_spectrum",
        "judge_file_sha256": hashlib.sha256(Path(judges.__file__).read_bytes()).hexdigest(),
        "salt_commitment_sha256": hashlib.sha256(salt.encode()).hexdigest(),
        "held_out_units": 12,
        "development_units": 8,
        "unit_kind": UNIT_KIND,
        "pass_rate_best_trivial_answer": best_trivial_rate,
        "best_trivial_answer": "trivial:" + best_trivial,
        "pass_rate_all_trivials": trivial_rates,
        "pass_rate_public_baseline_pwm_core": methods["baseline_pwm_core"]["pass_rate"],
        "pass_rate_best_fixed_setting_chosen_on_dev": methods["best_fixed(setting chosen on dev)"]["pass_rate"],
        "best_fixed_setting": best_setting,
        "pass_rate_adaptive_reference_solver": methods["adaptive_reference_solver"]["pass_rate"],
        "reference_curve_tolerance": "relative L2 <= 0.10 over 2800 mass-to-charge bins",
        "data": data_manifest,
        "hidden_key_manifest_sha256": hashlib.sha256(json.dumps(key_manifest, sort_keys=True).encode()).hexdigest(),
        "key_storage": str(keydir),
        "domain_expert_review": "QUEUED, not done",
        "date": time.strftime("%Y-%m-%d"),
    }
    _write(RESULT_DIR / "simulate.json", result)
    _write(RESULT_DIR / "simulate.version.json", version)
    provenance_path = ROOT / "results" / "IPWM-24" / "data_provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    provenance["atom_probe"] = data_manifest
    _write(provenance_path, provenance)
    return result, version


if __name__ == "__main__":
    _, version = build_and_evaluate()
    print(json.dumps({k: version[k] for k in ("pass_rate_best_trivial_answer",
        "pass_rate_public_baseline_pwm_core", "pass_rate_best_fixed_setting_chosen_on_dev",
        "pass_rate_adaptive_reference_solver")}, indent=2))
