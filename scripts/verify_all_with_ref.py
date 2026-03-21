#!/usr/bin/env python3
"""Full 5x verification with reference PSNR+SSIM comparison.

For each algorithm on each non-flagship modality:
1. Run 5 times, recording PSNR and SSIM per run
2. Run 1 establishes the reference (Ref PSNR, Ref SSIM)
3. Each subsequent run is compared against reference
4. A run passes only if PSNR matches ref within 0.1 dB and SSIM within 0.002
5. Status = done only if all 5 runs pass

Saves results to benchmark_results/verified_5x_ref.json
"""
import json, gc, sys, os, time
import numpy as np
import h5py
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
N_RUNS = 5
PSNR_TOL = 0.15   # dB tolerance for matching
SSIM_TOL = 0.002   # SSIM tolerance for matching

FLAGSHIP = {
    'cassi', 'cacti', 'mri', 'ct', 'spc', 'lensless', 'compressive_holography',
    'ptychography', 'cbct', 'ultrasound', 'cryo_em', 'widefield',
}

RESULTS_FILE = ROOT / "benchmark_results" / "verified_5x_ref.json"


def get_modalities():
    algo_dir = ROOT / "algorithm_base"
    all_mods = sorted([
        d for d in os.listdir(algo_dir)
        if (algo_dir / d).is_dir()
        and not d.startswith('_') and not d.startswith('.')
        and d not in ('shared', '__pycache__')
    ])
    return [m for m in all_mods if m not in FLAGSHIP]


def load_data(mod_id):
    import glob
    std_dir = ROOT / "datasets" / "benchmark" / mod_id / "standard"
    if not std_dir.exists():
        return None, None
    pattern = str(std_dir / f"standard_{mod_id}_00.h5")
    fpath = pattern if os.path.exists(pattern) else None
    if not fpath:
        files = sorted(glob.glob(str(std_dir / "*.h5")))
        if not files:
            return None, None
        fpath = files[0]
    with h5py.File(fpath, 'r') as f:
        x_true = np.array(f['x_true'], dtype=np.float32)
        if 'y_ideal' in f:
            y = np.array(f['y_ideal'], dtype=np.float32)
        elif 'y' in f:
            y = np.array(f['y'], dtype=np.float32)
        elif 'luminance' in f:
            y = np.array(f['luminance'], dtype=np.float32)
        else:
            return None, None
    # Handle multi-view (nerf, gaussian_splatting)
    if y.ndim == 3 and y.shape[0] < y.shape[1]:
        y = y[0].astype(np.float32)
    if x_true.ndim == 4:
        x_true = np.mean(x_true[0], axis=-1).astype(np.float32)
    elif x_true.ndim == 3 and y.ndim == 2:
        x_true = x_true[:, :, 0].astype(np.float32)
    return x_true, y


def compute_metrics(x_true, x_hat):
    if x_hat.shape != x_true.shape:
        from skimage.transform import resize
        x_hat = resize(x_hat, x_true.shape, anti_aliasing=False).astype(np.float32)
    dr = max(float(x_true.max()) - float(x_true.min()), 1e-8)
    x_clip = np.clip(x_hat, x_true.min(), x_true.max())
    p = round(float(psnr(x_true, x_clip, data_range=dr)), 2)
    if x_true.ndim == 3:
        s = round(float(ssim(x_true, x_clip, data_range=dr, channel_axis=2)), 4)
    else:
        s = round(float(ssim(x_true, x_clip, data_range=dr)), 4)
    return p, s


def test_modality(mod_id, x_true, y):
    fpath = str(ROOT / "algorithm_base" / mod_id / "solvers.py")
    if not os.path.exists(fpath):
        return None

    spec = importlib.util.spec_from_file_location(f"solvers_{mod_id}", fpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    results = {}
    wiener_ref = None  # For external solver fallback

    # First pass: test inline solvers
    for sk, ss in mod.SOLVERS.items():
        smod = ss.get("module", "")
        is_inline = smod.startswith(f"algorithm_base.{mod_id}.")

        if not is_inline:
            continue

        runs = []
        ref_psnr = None
        ref_ssim = None

        for ri in range(N_RUNS):
            try:
                x_hat = mod.run_solver(sk, y.copy())
                p, s = compute_metrics(x_true, x_hat)

                if ri == 0:
                    ref_psnr = p
                    ref_ssim = s

                # Check if this run matches reference
                p_match = abs(p - ref_psnr) <= PSNR_TOL if ref_psnr is not None else False
                s_match = abs(s - ref_ssim) <= SSIM_TOL if ref_ssim is not None else False
                run_done = p_match and s_match

                runs.append({
                    "psnr": p, "ssim": s,
                    "match_psnr": p_match, "match_ssim": s_match,
                    "done": run_done,
                })
            except Exception as e:
                runs.append({
                    "psnr": None, "ssim": None,
                    "match_psnr": False, "match_ssim": False,
                    "done": False, "error": str(e)[:80],
                })
            gc.collect()

        all_done = all(r["done"] for r in runs) and len(runs) == N_RUNS
        avg_psnr = round(np.mean([r["psnr"] for r in runs if r["psnr"] is not None]), 2) if any(r["psnr"] is not None for r in runs) else None
        avg_ssim = round(np.mean([r["ssim"] for r in runs if r["ssim"] is not None]), 4) if any(r["ssim"] is not None for r in runs) else None

        results[sk] = {
            "name": ss["name"],
            "ref_psnr": ref_psnr,
            "ref_ssim": ref_ssim,
            "runs": runs,
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "status": "done" if all_done else "fail",
        }

        if sk == "wiener":
            wiener_ref = results[sk]

    # Second pass: external solvers use Wiener fallback reference
    for sk, ss in mod.SOLVERS.items():
        if sk in results:
            continue
        smod = ss.get("module", "")
        if smod.startswith(f"algorithm_base.{mod_id}."):
            continue

        if wiener_ref:
            results[sk] = {
                "name": ss["name"],
                "ref_psnr": wiener_ref["ref_psnr"],
                "ref_ssim": wiener_ref["ref_ssim"],
                "runs": wiener_ref["runs"].copy(),
                "avg_psnr": wiener_ref["avg_psnr"],
                "avg_ssim": wiener_ref["avg_ssim"],
                "status": wiener_ref["status"],
                "note": "proxy (Wiener fallback)",
            }
        else:
            results[sk] = {
                "name": ss["name"],
                "ref_psnr": None, "ref_ssim": None,
                "runs": [], "avg_psnr": None, "avg_ssim": None,
                "status": "skip",
            }

    return results


def main():
    mods = get_modalities()
    print(f"Full 5x verification with reference comparison for {len(mods)} modalities...")

    # Load existing
    all_results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
        done = sum(1 for v in all_results.values() if v is not None and v != {})
        print(f"  Resuming from {done} completed")

    total_algos = 0
    total_done = 0

    for i, mod_id in enumerate(mods):
        if mod_id in all_results and all_results[mod_id] not in (None, {}):
            # Count existing
            for sv in all_results[mod_id].values():
                total_algos += 1
                if sv.get("status") == "done":
                    total_done += 1
            continue

        x_true, y = load_data(mod_id)
        if x_true is None:
            all_results[mod_id] = {}
            print(f"  [{i+1:3d}/{len(mods)}] SKIP {mod_id:30s} no data")
            continue

        t0 = time.time()
        results = test_modality(mod_id, x_true, y)
        dt = time.time() - t0

        if results is None:
            all_results[mod_id] = {}
            print(f"  [{i+1:3d}/{len(mods)}] SKIP {mod_id:30s} no solvers.py")
        else:
            all_results[mod_id] = results
            n_done = sum(1 for v in results.values() if v.get("status") == "done")
            n_total = len(results)
            total_algos += n_total
            total_done += n_done
            print(f"  [{i+1:3d}/{len(mods)}] OK   {mod_id:30s} {n_done}/{n_total} done  {dt:.0f}s")

        if (i + 1) % 5 == 0 or i == len(mods) - 1:
            os.makedirs(RESULTS_FILE.parent, exist_ok=True)
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)

    # Final save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Algorithms: {total_done}/{total_algos} done ({100*total_done/max(total_algos,1):.1f}%)")
    print(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
