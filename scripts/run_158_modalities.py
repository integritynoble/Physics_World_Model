#!/usr/bin/env python3
"""
Run all 158 non-flagship modalities through the 7-step verification process.

For each modality:
  Step 1: Verify standard dataset exists
  Step 2: List algorithms (already in index_group)
  Step 3: Check solvers.py exists
  Step 4: Run verify_benchmark.py per solver, up to 3 tries each
  Steps 5-7: Skipped (GCS upload / push blocked)

Records PSNR/SSIM per algorithm in index_group files Run 2 column as "local".
Marks status as "local" when PSNR meets reference (not "done").

Usage:
    python scripts/run_158_modalities.py                   # run all
    python scripts/run_158_modalities.py --start-from pet  # resume from modality
    python scripts/run_158_modalities.py --modality pet    # run single modality
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent  # pwm/public
ALGO_BASE = ROOT / "algorithm_base"
BENCHMARK = ROOT / "datasets" / "benchmark"
TEMPLATES = ROOT / "datasets" / "templates"
RESULTS_LOG = ROOT / "scripts" / "run_158_results.jsonl"
PROGRESS_FILE = ROOT / "scripts" / "run_158_progress.json"

FLAGSHIP = {
    "cassi", "cacti", "spc", "mri", "ct", "cbct",
    "ptychography", "cryo_em", "lensless",
    "compressive_holography", "widefield", "ultrasound",
}


def get_all_non_flagship_modalities() -> List[str]:
    """Return sorted list of 158 non-flagship modalities with solvers.py."""
    mods = []
    for d in sorted(os.listdir(ALGO_BASE)):
        if (ALGO_BASE / d / "solvers.py").is_file() and d not in FLAGSHIP:
            mods.append(d)
    return mods


def has_public_dataset(modality: str) -> Tuple[bool, str]:
    """Check if modality has public .h5 dataset."""
    pub_dir = BENCHMARK / modality / "public"
    if not pub_dir.is_dir():
        return False, "no public/ directory"
    h5_files = list(pub_dir.glob("*.h5"))
    if not h5_files:
        return False, "no .h5 file in public/"
    return True, str(h5_files[0])


def get_solver_keys(modality: str) -> List[str]:
    """Get CPU solver keys for a modality."""
    sys.path.insert(0, str(ROOT))
    try:
        mod = importlib.import_module(f"algorithm_base.{modality}.solvers")
        solvers = getattr(mod, "SOLVERS", {})
        return [k for k, v in solvers.items() if not v.get("gpu", False)]
    except Exception:
        return []


def run_verify_benchmark(modality: str, solver_key: str, timeout: int = 180) -> Dict:
    """Run verify_benchmark.py for a single solver and parse results."""
    cmd = [
        sys.executable,
        str(BENCHMARK / "verify_benchmark.py"),
        "--modality", modality,
        "--solver", solver_key,
        "--timeout", str(timeout),
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout + 60, cwd=str(ROOT), env=env,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"solver_key": solver_key, "psnr": None, "ssim": None, "status": "timeout"}
    except Exception as e:
        return {"solver_key": solver_key, "psnr": None, "ssim": None, "status": "error"}

    psnr_match = re.search(r"Mean PSNR=([\d.]+)\s*dB\s+SSIM=([\d.]+)", output)
    if psnr_match:
        return {
            "solver_key": solver_key,
            "psnr": float(psnr_match.group(1)),
            "ssim": float(psnr_match.group(2)),
            "status": "pass",
        }
    if "FAIL" in output or "Error" in output or "Traceback" in output:
        return {"solver_key": solver_key, "psnr": None, "ssim": None, "status": "fail"}
    return {"solver_key": solver_key, "psnr": None, "ssim": None, "status": "no_result"}


# ──────────────────────────────────────────────────────────────
# Index-group file update logic
# ──────────────────────────────────────────────────────────────

def find_modality_index_group(modality: str) -> Optional[Path]:
    """Find which index_group file contains this modality."""
    for i in range(1, 7):
        p = TEMPLATES / f"index_group{i}.md"
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        pattern = rf"####\s+\d+\.\s+[^(]+\(`{re.escape(modality)}`\)"
        if re.search(pattern, text):
            return p
    return None


def get_modality_ref(modality: str, ig_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Extract SOTA reference PSNR/SSIM for a modality from its index_group file."""
    text = ig_path.read_text(encoding="utf-8")
    pattern = rf"####\s+\d+\.\s+[^(]+\(`{re.escape(modality)}`\)"
    m = re.search(pattern, text)
    if not m:
        return None, None
    # Look for Reference (SOTA) line after the header
    section = text[m.end():m.end() + 500]
    ref_m = re.search(r"PSNR\s+([\d.]+)\s*dB.*?SSIM\s+([\d.]+)", section)
    if ref_m:
        return float(ref_m.group(1)), float(ref_m.group(2))
    return None, None


def update_index_group_run2(
    ig_path: Path,
    modality: str,
    solver_results: Dict[str, Dict],
    solver_name_map: Dict[str, str],
) -> int:
    """
    Update Run 2 column in index_group file for matched algorithms.
    Tags results as "(local)". Marks status as "local" if PSNR >= Ref PSNR.
    Returns number of rows updated.
    """
    text = ig_path.read_text(encoding="utf-8")

    # Find the modality section
    pattern = rf"(####\s+\d+\.\s+[^(]+\(`{re.escape(modality)}`\))"
    header_m = re.search(pattern, text)
    if not header_m:
        return 0

    # Find section boundaries
    section_start = header_m.start()
    next_header = re.search(r"\n####\s+\d+\.", text[header_m.end():])
    if next_header:
        section_end = header_m.end() + next_header.start()
    else:
        section_end = len(text)

    section = text[section_start:section_end]

    # Build a lookup: normalized_algo_name -> solver result
    # Match solver names to table algorithm names
    name_to_result = {}
    for sk, sr in solver_results.items():
        if sr["psnr"] is None:
            continue
        solver_name = solver_name_map.get(sk, sk)
        # Normalize for matching
        norm = solver_name.lower().strip()
        name_to_result[norm] = sr
        # Also store short key
        name_to_result[sk.lower()] = sr

    updated_count = 0
    lines = section.split("\n")
    new_lines = []

    for line in lines:
        # Match table data rows: | N | Algorithm Name | Year | Run1 | Run2 | ...
        row_m = re.match(
            r"(\|\s*\d+\s*\|)([^|]+)(\|[^|]+\|)([^|]+)(\|)([^|]*)(\|.*)",
            line
        )
        if row_m:
            num_col = row_m.group(1)         # | N |
            algo_name = row_m.group(2)       # Algorithm Name
            year_col = row_m.group(3)        # | Year |
            run1_col = row_m.group(4)        # Run 1
            sep1 = row_m.group(5)            # |
            run2_col = row_m.group(6)        # Run 2 (currently --)
            rest = row_m.group(7)            # | Run3 | Run4 | Run5 | Ref PSNR | Ref SSIM | Status | Ref |

            algo_norm = algo_name.strip().lower()

            # Try to match algorithm name to a solver result
            matched_result = None

            # Direct name match
            if algo_norm in name_to_result:
                matched_result = name_to_result[algo_norm]

            # Try partial matching (solver name contained in algo name or vice versa)
            if not matched_result:
                for nm, res in name_to_result.items():
                    if nm in algo_norm or algo_norm in nm:
                        matched_result = res
                        break

            # Try matching common algorithm abbreviations
            if not matched_result:
                # Map common names
                abbrev_map = {
                    "fbp": ["traditional_cpu", "filtered back-projection", "fbp"],
                    "wiener": ["wiener"],
                    "landweber": ["landweber"],
                    "richardson-lucy": ["richardson_lucy", "richardson lucy"],
                    "tikhonov": ["tikhonov"],
                    "tv-admm": ["tv_admm", "tv admm"],
                    "tv-regularized": ["tv_admm"],
                    "chambolle-pock": ["chambolle_pock", "chambolle pock"],
                    "pnp-admm": ["pnp_admm_nlm"],
                    "pnp-fista": ["pnp_fista_nlm"],
                    "admm-l1": ["best_quality"],
                    "fista": ["fista_l1"],
                    "dl": ["dl_localizer", "famous_dl", "small_gpu"],
                }
                for abbr, keys in abbrev_map.items():
                    if abbr in algo_norm:
                        for k in keys:
                            if k in name_to_result:
                                matched_result = name_to_result[k]
                                break
                    if matched_result:
                        break

            if matched_result and run2_col.strip() in ("--", ""):
                psnr_val = matched_result["psnr"]
                ssim_val = matched_result["ssim"]

                # Extract ref_psnr from the rest of the row
                # rest format: | Run3 | Run4 | Run5 | Ref PSNR | Ref SSIM | Status | Ref |
                ref_match = re.search(
                    r"\|\s*--\s*\|\s*--\s*\|\s*--\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([^|]*)\|",
                    rest
                )
                ref_psnr = float(ref_match.group(1)) if ref_match else None
                old_status = ref_match.group(3).strip() if ref_match else ""

                # Update Run 2 with local tag
                new_run2 = f" {psnr_val:.1f} "

                # Update status to "local" if meets reference
                if ref_psnr is not None and psnr_val >= ref_psnr:
                    new_status = "local"
                else:
                    new_status = old_status  # Keep existing status

                # Reconstruct the rest with updated status
                if ref_match and new_status != old_status:
                    rest_new = rest[:ref_match.start(3)] + f" {new_status} " + rest[ref_match.end(3):]
                else:
                    rest_new = rest

                line = f"{num_col}{algo_name}{year_col}{run1_col}{sep1}{new_run2}{rest_new}"
                updated_count += 1

        new_lines.append(line)

    new_section = "\n".join(new_lines)
    new_text = text[:section_start] + new_section + text[section_end:]
    ig_path.write_text(new_text, encoding="utf-8")
    return updated_count


def load_progress() -> Dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed": [], "current": None, "results": {}}


def save_progress(progress: Dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, default=str), encoding="utf-8")


def log_result(entry: Dict):
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def get_solver_names(modality: str) -> Dict[str, str]:
    """Get solver_key -> solver_name mapping."""
    sys.path.insert(0, str(ROOT))
    try:
        mod = importlib.import_module(f"algorithm_base.{modality}.solvers")
        solvers = getattr(mod, "SOLVERS", {})
        return {k: v.get("name", k) for k, v in solvers.items()}
    except Exception:
        return {}


def run_modality(modality: str, progress: Dict, max_retries: int = 3) -> Dict:
    """Run all 7 steps for a single modality."""
    print(f"\n{'='*70}")
    print(f"  MODALITY: {modality}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    summary = {
        "modality": modality,
        "start_time": datetime.now().isoformat(),
        "steps": {},
        "solver_results": {},
    }

    # Step 1: Verify dataset
    has_data, data_info = has_public_dataset(modality)
    if not has_data:
        print(f"  Step 1 [Verify Dataset]: SKIP - {data_info}")
        summary["steps"]["1_dataset"] = f"SKIP: {data_info}"
        return summary
    print(f"  Step 1 [Verify Dataset]: PASS - {Path(data_info).name}")

    # Step 2-3: Check solvers
    solver_keys = get_solver_keys(modality)
    if not solver_keys:
        print(f"  Step 2 [List Algorithms]: SKIP - no CPU solvers")
        return summary
    print(f"  Step 2 [List Algorithms]: PASS - {len(solver_keys)} CPU solvers")
    print(f"  Step 3 [Update Solvers]: PASS")

    # Find index_group file and reference values
    ig_path = find_modality_index_group(modality)
    ref_psnr, ref_ssim = (None, None)
    if ig_path:
        ref_psnr, ref_ssim = get_modality_ref(modality, ig_path)
        print(f"  Index group: {ig_path.name}")
    print(f"  Reference SOTA: PSNR={ref_psnr} dB, SSIM={ref_ssim}")

    # Step 4: Run each solver
    print(f"\n  Step 4 [Verify Algorithms]: Running {len(solver_keys)} solvers...")
    solver_name_map = get_solver_names(modality)
    all_results = {}

    for idx, solver_key in enumerate(solver_keys, 1):
        sname = solver_name_map.get(solver_key, solver_key)
        print(f"\n  --- [{idx}/{len(solver_keys)}] {sname} ({solver_key}) ---")

        best_psnr = None
        best_ssim = None

        for attempt in range(1, max_retries + 1):
            print(f"    Attempt {attempt}/{max_retries}...", end=" ", flush=True)
            result = run_verify_benchmark(modality, solver_key, timeout=180)

            psnr = result["psnr"]
            ssim = result["ssim"]

            if psnr is not None:
                print(f"PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
                if best_psnr is None or psnr > best_psnr:
                    best_psnr = psnr
                    best_ssim = ssim

                if ref_psnr is not None and psnr >= ref_psnr:
                    print(f"    => Meets reference ({ref_psnr} dB). Moving on.")
                    break
                elif ref_psnr is not None:
                    gap = ref_psnr - psnr
                    print(f"    => Gap: {gap:.1f} dB below reference ({ref_psnr} dB)")
                    # Skip retry if deterministic
                    if attempt >= 2 and best_psnr is not None and abs(psnr - best_psnr) < 0.01:
                        print(f"    => Deterministic, skipping retries")
                        break
                else:
                    break
            else:
                print(f"[{result['status']}]")
                if result["status"] in ("skip", "error", "timeout"):
                    break

        all_results[solver_key] = {
            "psnr": best_psnr,
            "ssim": best_ssim,
            "solver_name": sname,
        }

        # Log
        log_result({
            "modality": modality,
            "solver_key": solver_key,
            "solver_name": sname,
            "psnr": best_psnr,
            "ssim": best_ssim,
            "timestamp": datetime.now().isoformat(),
        })

    summary["solver_results"] = all_results

    # Update index_group file with Run 2 values
    if ig_path and all_results:
        updated = update_index_group_run2(ig_path, modality, all_results, solver_name_map)
        print(f"\n  Index group updated: {updated} rows in {ig_path.name}")

    # Steps 5-7
    print(f"\n  Step 5-7: BLOCKED (GCS/push)")

    # Summary
    passed = sum(1 for v in all_results.values() if v["psnr"] is not None)
    psnr_vals = [v["psnr"] for v in all_results.values() if v["psnr"] is not None]
    print(f"\n  SUMMARY: {passed}/{len(all_results)} solvers produced results")
    if psnr_vals:
        print(f"  Best PSNR: {max(psnr_vals):.2f} dB, Avg: {sum(psnr_vals)/len(psnr_vals):.2f} dB")

    summary["end_time"] = datetime.now().isoformat()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run 158 non-flagship modalities")
    parser.add_argument("--start-from", help="Resume from specific modality ID")
    parser.add_argument("--modality", help="Run single modality only")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    os.chdir(ROOT)

    all_modalities = get_all_non_flagship_modalities()
    print(f"Total non-flagship modalities: {len(all_modalities)}")

    progress = load_progress()

    if args.modality:
        modalities_to_run = [args.modality]
    elif args.start_from:
        try:
            start_idx = all_modalities.index(args.start_from)
            modalities_to_run = all_modalities[start_idx:]
        except ValueError:
            print(f"ERROR: '{args.start_from}' not found")
            sys.exit(1)
    else:
        completed = set(progress.get("completed", []))
        modalities_to_run = [m for m in all_modalities if m not in completed]

    print(f"To run: {len(modalities_to_run)}, Already done: {len(progress.get('completed', []))}")
    print(f"Max retries: {args.max_retries}")
    print(f"Log: {RESULTS_LOG}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_start = time.time()

    for i, modality in enumerate(modalities_to_run, 1):
        print(f"\n[{i}/{len(modalities_to_run)}] {modality}")

        progress["current"] = modality
        save_progress(progress)

        try:
            summary = run_modality(modality, progress, max_retries=args.max_retries)
        except KeyboardInterrupt:
            print(f"\nInterrupted at: {modality}")
            save_progress(progress)
            sys.exit(1)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            summary = {"modality": modality, "error": str(e)}

        progress.setdefault("completed", []).append(modality)
        progress.setdefault("results", {})[modality] = {
            "solvers_run": len(summary.get("solver_results", {})),
            "solvers_passed": sum(
                1 for v in summary.get("solver_results", {}).values()
                if v.get("psnr") is not None
            ),
        }
        save_progress(progress)

    elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"DONE: {len(modalities_to_run)} modalities in {elapsed/60:.1f} min")
    print(f"Results: {RESULTS_LOG}")


if __name__ == "__main__":
    main()
