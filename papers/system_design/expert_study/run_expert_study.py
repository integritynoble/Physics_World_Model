"""Run the prospective head-to-head expert study.

5 expert agents × 3 modalities = 15 runs, plus 3 Agent pipeline runs.
Each expert independently designs the forward model + reconstruction.

Usage:
    cd /home/spiritai/pwm/Physics_World_Model
    python papers/system_design/expert_study/run_expert_study.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import subprocess
import traceback
import inspect
from pathlib import Path
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from papers.system_design.expert_study.data_loader import load_data, get_design_brief
from papers.system_design.expert_study.expert_reconstructors import (
    EXPERT_DISPATCH,
    EXPERT_LOC,
    _adjoint_test,
    _radon_forward, _radon_adjoint,
    _mri_forward, _mri_adjoint,
    _cassi_forward, _cassi_adjoint,
)
from papers.system_design.expert_study.evaluate import compute_metrics

MODALITIES = ["ct", "mri", "sd_cassi"]
N_SAMPLES = 10
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXPERT_IDS = ["E1", "E2", "E3", "E4", "E5"]


def run_adjoint_validation(modality: str, samples: list[dict]) -> dict:
    """Run adjoint consistency test for each modality's forward model."""
    s = samples[0]
    if modality == "ct":
        img_size = s["x_true"].shape[0]
        angles = s["calibration"]["angles_rad"]
        n_det = s["measurements"].shape[1]
        A = lambda x: _radon_forward(x, angles, n_det)
        At = lambda y: _radon_adjoint(y, angles, img_size)
        err, passed = _adjoint_test(A, At, (img_size, img_size), (len(angles), n_det))
    elif modality == "mri":
        coil_maps = s["calibration"]["coil_maps"]
        mask = s["calibration"]["mask"]
        ny, nx = s["x_true"].shape
        A = lambda x: _mri_forward(x, coil_maps, mask)
        At = lambda y: _mri_adjoint(y, coil_maps, mask)
        err, passed = _adjoint_test(A, At, (ny, nx), coil_maps.shape, dtype=np.complex128)
    elif modality == "sd_cassi":
        mask = s["calibration"]["coded_aperture_mask"]
        n_bands = s["x_true"].shape[2]
        ny, nx = mask.shape
        # Use forward model's output shape (w + (n_bands-1)*step), not actual measurement shape
        step = 2
        w_meas_fwd = nx + (n_bands - 1) * step
        A = lambda x: _cassi_forward(x, mask)
        At = lambda y: _cassi_adjoint(y, mask, n_bands)
        err, passed = _adjoint_test(A, At, (ny, nx, n_bands), (ny, w_meas_fwd))
    else:
        return {"error": 0, "passed": False}

    return {"error": float(err), "passed": bool(passed)}


def run_single_expert(expert_id: str, modality: str,
                      samples: list[dict]) -> dict:
    """Run one expert on one modality."""
    func = EXPERT_DISPATCH.get((expert_id, modality))
    if func is None:
        return {"success": False, "error": "No implementation"}

    info = EXPERT_LOC[expert_id]

    t_start = time.perf_counter()
    try:
        recons = func(samples)
        t_end = time.perf_counter()
    except Exception as e:
        t_end = time.perf_counter()
        return {
            "agent_id": expert_id,
            "agent_name": info["name"],
            "philosophy": info["philosophy"],
            "modality": modality,
            "wall_clock_s": round(t_end - t_start, 2),
            "psnr_mean": 0.0,
            "psnr_std": 0.0,
            "ssim_mean": 0.0,
            "ssim_std": 0.0,
            "n_samples_success": 0,
            "lines_of_code": info["loc"].get(modality, 0),
            "success": False,
            "error": str(e)[:500],
        }

    # Compute metrics
    psnrs, ssims = [], []
    for i, (x_hat, sample) in enumerate(zip(recons, samples)):
        m = compute_metrics(x_hat, sample["x_true"])
        psnrs.append(m["psnr"])
        ssims.append(m["ssim"])

    wall_clock = t_end - t_start

    return {
        "agent_id": expert_id,
        "agent_name": info["name"],
        "philosophy": info["philosophy"],
        "modality": modality,
        "wall_clock_s": round(wall_clock, 2),
        "psnr_mean": round(float(np.mean(psnrs)), 2),
        "psnr_std": round(float(np.std(psnrs)), 2),
        "ssim_mean": round(float(np.mean(ssims)), 4),
        "ssim_std": round(float(np.std(ssims)), 4),
        "n_samples_success": len(psnrs),
        "lines_of_code": info["loc"].get(modality, 0),
        "success": True,
        "error": None,
    }


def run_agent_pipeline(modality: str, samples: list[dict]) -> dict:
    """Run the existing Plan→Judge→Performance pipeline for comparison.

    Uses the existing main.py orchestrator with the same design brief.
    """
    design_brief = get_design_brief(modality)
    mod_map = {"sd_cassi": "cassi", "ct": "ct", "mri": "mri"}
    mod_name = mod_map.get(modality, modality)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        meas_path = tmpdir / "measurements.npy"
        xtrue_path = tmpdir / "x_true.npy"
        np.save(meas_path, samples[0]["measurements"])
        np.save(xtrue_path, samples[0]["x_true"])

        t_start = time.perf_counter()

        # Forward period
        cmd_fwd = [
            sys.executable,
            str(ROOT / "papers" / "system_design" / "main.py"),
            "--modality", mod_name,
            "--period", "forward",
            "--prompt", design_brief,
        ]
        env = {**os.environ, "PYTHONPATH": str(ROOT / "packages" / "pwm_core")}

        try:
            proc_fwd = subprocess.run(
                cmd_fwd, capture_output=True, text=True, timeout=300,
                cwd=str(ROOT), env=env,
            )
        except (subprocess.TimeoutExpired, Exception) as e:
            t_end = time.perf_counter()
            return _pipeline_result(modality, t_end - t_start, False,
                                    error=f"Forward: {e}")

        # Reconstruction period
        cmd_recon = [
            sys.executable,
            str(ROOT / "papers" / "system_design" / "main.py"),
            "--modality", mod_name,
            "--period", "reconstruction",
            "--prompt", design_brief,
            "--measurements", str(meas_path),
            "--x_true", str(xtrue_path),
        ]

        try:
            proc_recon = subprocess.run(
                cmd_recon, capture_output=True, text=True, timeout=300,
                cwd=str(ROOT), env=env,
            )
        except (subprocess.TimeoutExpired, Exception) as e:
            t_end = time.perf_counter()
            return _pipeline_result(modality, t_end - t_start, False,
                                    error=f"Recon: {e}")

        t_end = time.perf_counter()

    # Parse metrics from stdout
    import re
    psnr_val = None
    ssim_val = None
    for line in (proc_recon.stdout + proc_fwd.stdout).split("\n"):
        m = re.search(r"psnr\s*[=:]\s*([\d.]+)", line, re.I)
        if m:
            psnr_val = float(m.group(1))
        m = re.search(r"ssim\s*[=:]\s*([\d.]+)", line, re.I)
        if m:
            ssim_val = float(m.group(1))

    # Count spec LoC from output files
    outputs_dir = ROOT / "papers" / "system_design" / "outputs"
    spec_files = sorted(outputs_dir.glob(f"{mod_name}_*.md"))
    spec_loc = 0
    if spec_files:
        spec_loc = sum(1 for l in spec_files[-1].read_text().split("\n")
                       if l.strip() and not l.strip().startswith("#"))

    return _pipeline_result(
        modality, t_end - t_start,
        success=psnr_val is not None,
        psnr=psnr_val or 0.0,
        ssim=ssim_val or 0.0,
        loc=spec_loc,
        error=(proc_recon.stderr[-300:] if psnr_val is None else None),
    )


def _pipeline_result(modality, time_s, success, psnr=0.0, ssim=0.0,
                     loc=0, error=None):
    return {
        "agent_id": "Pipeline",
        "agent_name": "Plan→Judge→Performance",
        "philosophy": "3-Agent LLM pipeline",
        "modality": modality,
        "wall_clock_s": round(time_s, 2),
        "psnr_mean": round(psnr, 2),
        "ssim_mean": round(ssim, 4),
        "lines_of_code": loc,
        "success": success,
        "error": error,
    }


def print_results(all_results: list[dict]):
    """Print formatted results table."""
    print("\n" + "=" * 95)
    print("PROSPECTIVE HEAD-TO-HEAD EXPERT STUDY RESULTS")
    print("=" * 95)

    # Detail table
    print(f"\n{'Agent':<10} {'Philosophy':<18} {'Modality':<12} "
          f"{'PSNR(dB)':>10} {'SSIM':>8} {'Time(s)':>8} {'LoC':>5} {'OK?':<5}")
    print("-" * 80)

    for mod in MODALITIES:
        mod_results = [r for r in all_results if r["modality"] == mod]
        for r in sorted(mod_results, key=lambda x: x["agent_id"]):
            if r["success"]:
                psnr_s = f"{r['psnr_mean']:.1f}±{r.get('psnr_std',0):.1f}"
                ssim_s = f"{r['ssim_mean']:.3f}"
                loc_s = str(r["lines_of_code"])
            else:
                psnr_s = "--"
                ssim_s = "--"
                loc_s = "--"
            print(f"{r['agent_id']:<10} {r.get('philosophy',''):<18} "
                  f"{r['modality']:<12} {psnr_s:>10} {ssim_s:>8} "
                  f"{r['wall_clock_s']:>8.1f} {loc_s:>5} "
                  f"{'Y' if r['success'] else 'N':<5}")
        print()

    # Summary per modality
    print("\n--- AGGREGATE SUMMARY ---")
    print(f"{'Modality':<12} {'Expert PSNR':>14} {'Expert SSIM':>12} "
          f"{'Expert Time':>12} {'Pipeline':>10} {'Q-ratio':>8}")
    print("-" * 72)

    for mod in MODALITIES:
        experts = [r for r in all_results
                   if r["modality"] == mod and r["agent_id"] != "Pipeline"
                   and r["success"]]
        pipe = [r for r in all_results
                if r["modality"] == mod and r["agent_id"] == "Pipeline"]

        if experts:
            ep = np.mean([r["psnr_mean"] for r in experts])
            es = np.mean([r["ssim_mean"] for r in experts])
            et = np.mean([r["wall_clock_s"] for r in experts])
            ep_std = np.std([r["psnr_mean"] for r in experts])

            pipe_psnr = pipe[0]["psnr_mean"] if pipe and pipe[0]["success"] else 0
            qr = pipe_psnr / ep * 100 if ep > 0 and pipe_psnr > 0 else 0

            print(f"{mod:<12} {ep:>6.1f}±{ep_std:<6.1f} {es:>11.3f} "
                  f"{et:>11.1f}s "
                  f"{pipe_psnr:>9.1f} {qr:>7.1f}%")
        else:
            print(f"{mod:<12} {'--':>14} {'--':>12} {'--':>12} {'--':>10} {'--':>8}")


def main():
    print(f"Prospective Expert Study — {datetime.now().isoformat()}")
    print(f"Modalities: {MODALITIES}")
    print(f"Experts: {EXPERT_IDS}")
    print(f"Samples per modality: {N_SAMPLES}")

    all_results = []

    for modality in MODALITIES:
        print(f"\n{'='*60}")
        print(f"MODALITY: {modality} — {get_design_brief(modality)[:60]}...")
        print(f"{'='*60}")

        # Load data
        print("  Loading data...")
        samples = load_data(modality, n_samples=N_SAMPLES)
        print(f"  Loaded {len(samples)} samples, "
              f"meas={samples[0]['measurements'].shape}")

        # Adjoint validation
        print("  Running adjoint validation...")
        adj = run_adjoint_validation(modality, samples)
        print(f"  Adjoint test: err={adj['error']:.2e}, "
              f"passed={'YES' if adj['passed'] else 'NO'}")

        # Run 5 experts
        for eid in EXPERT_IDS:
            info = EXPERT_LOC[eid]
            print(f"\n  --- {eid}: {info['name']} ({info['philosophy']}) ---")
            result = run_single_expert(eid, modality, samples)
            all_results.append(result)

            if result["success"]:
                print(f"  PSNR={result['psnr_mean']:.1f}±{result['psnr_std']:.1f} dB, "
                      f"SSIM={result['ssim_mean']:.3f}, "
                      f"Time={result['wall_clock_s']:.1f}s, "
                      f"LoC={result['lines_of_code']}")
            else:
                print(f"  FAILED: {result['error'][:100]}")

        # Save incremental
        _save_results(all_results)

    # Print summary
    print_results(all_results)

    # Save final
    _save_results(all_results)
    print(f"\nResults saved to {RESULTS_DIR}/expert_study_results.json")


def _save_results(all_results):
    path = RESULTS_DIR / "expert_study_results.json"
    with open(path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "modalities": MODALITIES,
            "n_samples": N_SAMPLES,
            "experts": EXPERT_IDS,
            "runs": all_results,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
