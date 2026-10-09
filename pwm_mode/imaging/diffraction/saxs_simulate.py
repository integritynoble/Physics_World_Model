"""Build and evaluate the SAXS measured-reference simulation judge."""
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

ROOT = Path(__file__).resolve().parents[3]
KEY_DIR = Path(os.environ.get("IPWM24_KEYS", "/tmp/ipwm24-keys"))
PUBLIC_DIR = Path(os.environ.get("IPWM24_PUBLIC", "/tmp/ipwm24-public"))
RESULT_DIR = ROOT / "results" / "IPWM-24" / "saxs"
CRITERION = {"relative_l2_max": 0.10}
UNIT_KIND = "detector frame (one-second SiO2 droplet exposure)"


def _salt() -> str:
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    p = KEY_DIR / "SALT"
    if not p.exists():
        p.write_text(secrets.token_hex(32) + "\n")
    return p.read_text().strip()


def _unit_id(salt: str, source_name: str) -> str:
    return hashlib.sha256((salt + "|saxs|simulate|" + source_name).encode()).hexdigest()[:16]


def _jitter_curve(curve: np.ndarray, salt: str, source_name: str) -> np.ndarray:
    """Add a small held-out detector-gain perturbation committed by the secret salt."""
    seed = int.from_bytes(hashlib.sha256((salt + "|gain|" + source_name).encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    gain = 1.0 + rng.uniform(-0.004, 0.004)
    offset = rng.uniform(-0.0002, 0.0002)
    return np.asarray(curve, dtype=np.float64) * gain + offset


def _metadata(meta: dict[str, Any], unit_id: str) -> dict[str, Any]:
    return {
        "unit": unit_id,
        "modality": "saxs",
        "sample": "SiO2 nanoparticles in microfluidic droplets",
        "exposure_s": meta["exposure_s"],
        "wavelength_m": meta["wavelength_m"],
        "pixel_size_m": meta["pixel_size_m"],
        "detector_center_px": meta["center_px"],
        "radial_coordinate": "detector radius (pixel)",
        "reference_curve_bins": 29,
    }


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_dev() -> tuple[list[dict[str, Any]], np.ndarray]:
    devdir = PUBLIC_DIR / "saxs" / "simulate" / "dev"
    rows = [json.loads(p.read_text()) for p in sorted(devdir.glob("*.json"))]
    curves = np.asarray([r["reference_curve"] for r in rows], dtype=np.float64)
    if curves.shape != (8, 29):
        raise RuntimeError(f"expected 8 public development curves, found {curves.shape}")
    return rows, curves


def _judge(sub, key, criterion=CRITERION):
    return judges.judge_saxs_simulate(sub, key, criterion)


def _rate(verdicts: list[dict[str, Any]]) -> float:
    return float(np.mean([v["pass"] for v in verdicts])) if verdicts else 0.0


def _candidate_mean(curves: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return curves.mean(axis=0) * scale


def build_and_evaluate() -> tuple[dict[str, Any], dict[str, Any]]:
    salt = _salt()
    source_manifest = data.fetch_saxs()
    devdir = PUBLIC_DIR / "saxs" / "simulate" / "dev"
    testdir = PUBLIC_DIR / "saxs" / "simulate" / "test"
    keytask = KEY_DIR / "saxs" / "simulate"
    devdir.mkdir(parents=True, exist_ok=True)
    testdir.mkdir(parents=True, exist_ok=True)
    keytask.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for i, filename in enumerate(data.SAXS_FILES):
        centers, measured = data.radial_curve(data.SAXS_DIR / filename)
        _, meta = data.read_edf(data.SAXS_DIR / filename)
        uid = _unit_id(salt, filename)
        reference = _jitter_curve(measured, salt, filename)
        all_rows.append({"filename": filename, "unit": uid, "centers": centers,
                         "meta": meta, "reference": reference, "split": "dev" if i < 8 else "test"})

    # Public development examples expose measured curves; held-out public inputs expose
    # acquisition metadata only. The hidden gain perturbation means a test key cannot be
    # reconstructed exactly from the listed metadata or from a copied development curve.
    dev_public_rows = []
    for row in all_rows[:8]:
        public = _metadata(row["meta"], row["unit"])
        public["reference_curve"] = row["reference"].tolist()
        out = devdir / f"{row['unit']}.json"
        data.write_json(out, public)
        dev_public_rows.append(public)
    test_public_rows = []
    for row in all_rows[8:]:
        public = _metadata(row["meta"], row["unit"])
        data.write_json(testdir / f"{row['unit']}.json", public)
        test_public_rows.append(public)

    forbidden = sorted({data.curve_sha256(r["reference_curve"]) for r in dev_public_rows})
    key_hashes = {}
    for row in all_rows[8:]:
        kpath = keytask / f"{row['unit']}.npz"
        _save_npz(kpath, reference_curve=row["reference"], radius_px=row["centers"],
                  forbidden_public_curve_sha256=np.asarray(forbidden, dtype="U64"))
        key_hashes[kpath.name] = hashlib.sha256(kpath.read_bytes()).hexdigest()

    dev_rows, dev_curves = _load_dev()
    test_keys = {}
    for row in all_rows[8:]:
        kpath = keytask / f"{row['unit']}.npz"
        with np.load(kpath, allow_pickle=False) as z:
            test_keys[row["unit"]] = {
                "reference_curve": z["reference_curve"],
                "forbidden_public_curve_sha256": z["forbidden_public_curve_sha256"].tolist(),
            }

    # Fixed scale is selected only on the development split, using leave-one-out predictions.
    settings = [{"scale": s} for s in (0.96, 0.98, 1.0, 1.02, 1.04)]
    setting_scores = []
    for setting in settings:
        verdicts = []
        for i, public in enumerate(dev_rows):
            train = np.delete(dev_curves, i, axis=0)
            pred = _candidate_mean(train, setting["scale"])
            key = {"reference_curve": dev_curves[i], "forbidden_public_curve_sha256": []}
            verdicts.append(_judge({"curve": pred}, key))
        setting_scores.append({"setting": setting, "dev_pass_rate": _rate(verdicts)})
    best_setting = max(setting_scores, key=lambda x: (x["dev_pass_rate"], -abs(x["setting"]["scale"] - 1.0)))["setting"]

    def result_method(name: str, fn) -> dict[str, Any]:
        verdicts = []
        for row in all_rows[8:]:
            sub = fn(row)
            verdicts.append(_judge(sub, test_keys[row["unit"]]))
        return {"pass_rate": _rate(verdicts), "n": len(verdicts),
                "verdicts": [{"unit": row["unit"], **verdict} for row, verdict in zip(all_rows[8:], verdicts)]}

    mean_curve = _candidate_mean(dev_curves, best_setting["scale"])
    blur_curve = gaussian_filter1d(_candidate_mean(dev_curves), sigma=1.5)
    const_curve = np.full(29, float(np.mean(dev_curves)))
    # A direct copy of any public training file is always rejected by the key fingerprint list.
    copy_curve = dev_curves[0].copy()

    methods = {
        "trivial:input_unchanged": result_method("input_unchanged", lambda row: row["meta"]),
        "trivial:constant": result_method("constant", lambda row: {"curve": const_curve}),
        "trivial:zeros": result_method("zeros", lambda row: {"curve": np.zeros(29)}),
        "trivial:blur": result_method("blur", lambda row: {"curve": blur_curve}),
        "trivial:copy_public": result_method("copy_public", lambda row: {"curve": copy_curve}),
        "best_fixed(setting chosen on dev)": result_method("best_fixed", lambda row: {"curve": mean_curve}),
        "adaptive_reference_solver": result_method("adaptive_reference_solver", lambda row: {
            "curve": _candidate_mean(dev_curves, best_setting["scale"] * row["meta"]["exposure_s"])}),
    }
    try:
        from pwm_mode.imaging import lifecycle
        from pwm_mode.imaging.lifecycle import test_object
        baseline_y = lifecycle.simulate("saxs", test_object("saxs", seed=42))["y"]
        methods["baseline_pwm_core"] = result_method("baseline", lambda row: {"curve": baseline_y})
    except Exception as exc:  # record a real baseline failure rather than silently omitting it
        methods["baseline_pwm_core"] = {"pass_rate": 0.0, "n": len(all_rows[8:]),
            "verdicts": [{"unit": row["unit"], "pass": False, "metrics": {},
                          "reasons": [f"pwm_core baseline unavailable: {type(exc).__name__}: {exc}"]}
                         for row in all_rows[8:]]}

    trivial_rates = {name.removeprefix("trivial:"): item["pass_rate"]
                     for name, item in methods.items() if name.startswith("trivial:")}
    best_trivial_name, best_trivial_rate = max(trivial_rates.items(), key=lambda kv: kv[1])
    result = {
        "methods": methods,
        "setting_scores_dev": setting_scores,
        "best_fixed_setting": best_setting,
        "development": {"n": 8, "unit_kind": UNIT_KIND},
        "held_out": {"n": 12, "unit_kind": UNIT_KIND},
        "limits": [
            "Reference curves are radial means from the 1 s SiO2 droplet frames; adjacent frames are repeated measurements, not independent specimens.",
            "Curve coordinate is detector radius in pixels. Absolute q is not scored because the source README says 1.786 m while EDF headers report 1.4 m.",
            "A secret per-frame gain/offset perturbation in the hidden key prevents exact recovery from public metadata or copied development curves.",
        ],
    }
    key_manifest = {name: digest for name, digest in sorted(key_hashes.items())}
    data_manifest = {
        "dataset": "Serial Small- and Wide-Angle X-ray Scattering with Laboratory Sources",
        "doi": "10.5518/1170",
        "url": "https://archive.researchdata.leeds.ac.uk/1010/",
        "license": "CC BY 4.0",
        "retained_files": source_manifest,
        "subset": "beam_0_00046.edf through beam_0_00065.edf; 20 one-second SiO2 nanoparticle droplet frames",
        "processed_unit": "radial mean intensity from radii 60..350 detector pixels, 29 bins; EDF Dummy pixels excluded",
    }
    version = {
        "task": "saxs/simulate",
        "criterion_version": judges.CRITERION_VERSION,
        "criterion": CRITERION,
        "judge_type": "saxs_simulation_curve",
        "judge_file_sha256": hashlib.sha256(Path(judges.__file__).read_bytes()).hexdigest(),
        "salt_commitment_sha256": hashlib.sha256(salt.encode()).hexdigest(),
        "held_out_units": 12,
        "development_units": 8,
        "unit_kind": UNIT_KIND,
        "pass_rate_best_trivial_answer": best_trivial_rate,
        "best_trivial_answer": "trivial:" + best_trivial_name,
        "pass_rate_all_trivials": trivial_rates,
        "pass_rate_public_baseline_pwm_core": methods["baseline_pwm_core"]["pass_rate"],
        "pass_rate_best_fixed_setting_chosen_on_dev": methods["best_fixed(setting chosen on dev)"]["pass_rate"],
        "best_fixed_setting": best_setting,
        "pass_rate_adaptive_reference_solver": methods["adaptive_reference_solver"]["pass_rate"],
        "reference_curve_tolerance": "relative L2 <= 0.10 over the 29 measured radial bins",
        "data": data_manifest,
        "hidden_key_manifest_sha256": hashlib.sha256(json.dumps(key_manifest, sort_keys=True).encode()).hexdigest(),
        "key_storage": str(keytask),
        "domain_expert_review": "QUEUED, not done",
        "date": time.strftime("%Y-%m-%d"),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data.write_json(RESULT_DIR / "simulate.json", result)
    data.write_json(RESULT_DIR / "simulate.version.json", version)
    provenance_path = ROOT / "results" / "IPWM-24" / "data_provenance.json"
    old = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    old["saxs"] = data_manifest
    data.write_json(provenance_path, old)
    return result, version


if __name__ == "__main__":
    res, ver = build_and_evaluate()
    print(json.dumps({"pass_rate_best_trivial_answer": ver["pass_rate_best_trivial_answer"],
                      "pass_rate_public_baseline_pwm_core": ver["pass_rate_public_baseline_pwm_core"],
                      "pass_rate_best_fixed_setting_chosen_on_dev": ver["pass_rate_best_fixed_setting_chosen_on_dev"],
                      "pass_rate_adaptive_reference_solver": ver["pass_rate_adaptive_reference_solver"]}, indent=2))

