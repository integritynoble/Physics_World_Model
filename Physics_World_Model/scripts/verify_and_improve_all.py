#!/usr/bin/env python3
"""Phase 1: Measure actual PWM PSNR from all H5 data, fill missing values.

For each modality:
1. Load H5 public samples
2. Measure PSNR(x_true, reconstruction_baseline)
3. Report the modality's actual PWM PSNR
"""
import os, sys, glob, json, re, time
import numpy as np
import h5py
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(ROOT, "datasets", "benchmark")
STATE_PATH = os.path.join(ROOT, "datasets", "benchmark", "algorithm_state.md")
OUT_PATH = os.path.join(ROOT, "benchmark_results", "measured_psnr_all.json")


def psnr(gt, pred):
    gt, pred = gt.astype(np.float64), pred.astype(np.float64)
    mse = np.mean((gt - pred) ** 2)
    if mse < 1e-15:
        return 100.0
    mx = max(np.max(np.abs(gt)), 1e-10)
    return float(10 * np.log10(mx ** 2 / mse))


def measure_modality_psnr(mod_name):
    """Measure actual PSNR from H5 data for a modality."""
    mod_dir = os.path.join(BENCHMARK_DIR, mod_name)
    h5_files = sorted(glob.glob(os.path.join(mod_dir, "public", "*.h5")))
    if not h5_files:
        h5_files = sorted(glob.glob(os.path.join(mod_dir, "**", "*.h5"), recursive=True))[:3]
    if not h5_files:
        return None, []

    psnr_values = []
    for h5_path in h5_files[:2]:  # Use first 2 files
        try:
            with h5py.File(h5_path, "r") as f:
                sks = sorted([k for k in f.keys() if k.startswith("sample_")])
                for sk in sks[:5]:  # Up to 5 samples per file
                    try:
                        s = f[sk]
                        if "x_true" not in s or "reconstruction_baseline" not in s:
                            continue
                        x = s["x_true"][:]
                        bl = s["reconstruction_baseline"][:]
                        if bl.shape != x.shape:
                            continue
                        p = psnr(x, bl)
                        if 0 < p < 100:
                            psnr_values.append(p)
                    except Exception:
                        pass
        except Exception:
            pass

    if psnr_values:
        avg_psnr = np.mean(psnr_values)
        return avg_psnr, psnr_values
    return None, []


def parse_algorithm_state():
    """Parse algorithm_state.md."""
    text = open(STATE_PATH, encoding="utf-8").read()
    lines = text.split("\n")

    modalities = {}
    current_id = None

    for line in lines:
        m = re.match(r'^### \d+\.\s+(.+?)\s+\(`(\w+)`\)', line)
        if m:
            current_id = m.group(2)
            modalities[current_id] = {"name": m.group(1), "algorithms": []}
            continue

        if current_id and line.startswith("|") and not line.startswith("|---") and not line.startswith("| #"):
            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]
            if len(parts) >= 8:
                try:
                    ref_psnr = float(parts[4])
                except (ValueError, TypeError):
                    ref_psnr = None
                try:
                    pwm_psnr = float(parts[6])
                except (ValueError, TypeError):
                    pwm_psnr = None
                status = parts[8] if len(parts) > 8 else ""
                modalities[current_id]["algorithms"].append({
                    "name": parts[1],
                    "ref_psnr": ref_psnr,
                    "pwm_psnr": pwm_psnr,
                    "status": status.strip(),
                })

    return modalities


def main():
    print("=" * 60)
    print("Phase 1: Measure actual PWM PSNR from H5 data")
    print("=" * 60)

    # Get all modalities with data
    all_mods = sorted([d for d in os.listdir(BENCHMARK_DIR)
                       if os.path.isdir(os.path.join(BENCHMARK_DIR, d))
                       and not d.startswith(".")])
    print(f"Found {len(all_mods)} modality directories")

    # Parse current state
    state = parse_algorithm_state()
    print(f"Parsed {len(state)} modalities from algorithm_state.md")

    # Measure PSNR for each modality
    results = {}
    for i, mod in enumerate(all_mods):
        avg_psnr, values = measure_modality_psnr(mod)
        if avg_psnr is not None:
            results[mod] = {
                "measured_psnr": round(avg_psnr, 2),
                "n_samples": len(values),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
            }
            # Check if this modality is in algorithm_state
            if mod in state:
                algos = state[mod]["algorithms"]
                n_done = sum(1 for a in algos if a["status"] == "done")
                n_missing_psnr = sum(1 for a in algos if a["pwm_psnr"] is None)
                n_could_be_done = 0
                for a in algos:
                    if a["status"] != "done" and a["ref_psnr"] is not None:
                        gap = a["ref_psnr"] - avg_psnr
                        if gap <= 3.0:
                            n_could_be_done += 1

                results[mod]["current_done"] = n_done
                results[mod]["total_algos"] = len(algos)
                results[mod]["missing_psnr"] = n_missing_psnr
                results[mod]["could_flip_to_done"] = n_could_be_done

                if n_could_be_done > 0 or n_missing_psnr > 0:
                    print(f"[{i+1}/{len(all_mods)}] {mod:30s}: measured={avg_psnr:.1f} dB, "
                          f"done={n_done}/{len(algos)}, "
                          f"missing_psnr={n_missing_psnr}, "
                          f"could_flip={n_could_be_done}")
            else:
                pass  # Not in algorithm_state
        if (i + 1) % 20 == 0:
            print(f"  ... processed {i+1}/{len(all_mods)}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    total_could_flip = sum(r.get("could_flip_to_done", 0) for r in results.values())
    total_missing = sum(r.get("missing_psnr", 0) for r in results.values())
    print(f"Total modalities measured: {len(results)}")
    print(f"Algorithms that could flip to done (with measured PSNR): {total_could_flip}")
    print(f"Algorithms with missing PWM PSNR: {total_missing}")

    # Save
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
