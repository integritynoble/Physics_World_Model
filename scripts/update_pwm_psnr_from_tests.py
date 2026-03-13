#!/usr/bin/env python3
"""Update algorithm_state.md with best available solver PSNR from test results.

Reads comprehensive_algorithm_test.json and updates PWM PSNR/SSIM in
algorithm_state.md to use the best solver result per modality.

Then applies done criteria: PWM_PSNR >= Ref_PSNR - 3.0 dB.
"""
import json, re, math
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATE_FILE = ROOT / "datasets" / "benchmark" / "algorithm_state.md"
TEST_FILE = ROOT / "benchmark_results" / "comprehensive_algorithm_test.json"
THRESHOLD = 3.0


def main():
    # Load test results
    test_data = json.load(open(TEST_FILE, encoding="utf-8"))
    mods = test_data["modalities"]

    # Extract best solver PSNR/SSIM per modality
    best_per_mod = {}
    for mod_name, mod_data in mods.items():
        if not isinstance(mod_data, dict) or "solvers" not in mod_data:
            continue
        solvers = mod_data["solvers"]
        best_psnr = -999
        best_ssim = 0
        best_solver = ""
        for sname, sdata in solvers.items():
            if not isinstance(sdata, dict):
                continue
            psnr = sdata.get("psnr_db")
            ssim = sdata.get("ssim", 0)
            if psnr is not None and isinstance(psnr, (int, float)):
                if not (math.isinf(psnr) or math.isnan(psnr)):
                    if psnr > best_psnr:
                        best_psnr = psnr
                        best_ssim = ssim if isinstance(ssim, (int, float)) and not math.isnan(ssim) else 0
                        best_solver = sname
        if best_psnr > -999:
            best_per_mod[mod_name] = {
                "psnr": round(best_psnr, 1),
                "ssim": round(best_ssim, 4),
                "solver": best_solver,
            }

    # Handle aliases
    if "sd_cassi" in best_per_mod and "cassi" not in best_per_mod:
        best_per_mod["cassi"] = best_per_mod["sd_cassi"]
    if "spc_kronecker" in best_per_mod and "spc" not in best_per_mod:
        best_per_mod["spc"] = best_per_mod["spc_kronecker"]

    # Read and update algorithm_state.md
    content = STATE_FILE.read_text(encoding="utf-8")
    lines = content.split("\n")

    current_mod = None
    updated_psnr = 0
    done_added = 0
    done_removed = 0
    done_kept = 0

    for i, line in enumerate(lines):
        m = re.match(r"^### \d+\.\s+.*\(`(\w+)`\)", line)
        if m:
            current_mod = m.group(1)
            continue

        if not (current_mod and line.startswith("|")):
            continue

        cols = line.split("|")
        if len(cols) < 10:
            continue
        try:
            int(cols[1].strip())
        except (ValueError, IndexError):
            continue

        algo_name = cols[2].strip()
        ref_str = cols[5].strip()
        old_pwm_str = cols[7].strip()
        old_ssim_str = cols[8].strip()
        old_status = cols[9].strip()

        # Update PWM PSNR with best solver
        if current_mod in best_per_mod:
            new_psnr = best_per_mod[current_mod]["psnr"]
            new_ssim = best_per_mod[current_mod]["ssim"]

            # Only update if better than current
            old_pwm = None
            try:
                old_pwm = float(old_pwm_str)
            except:
                pass

            if old_pwm is None or new_psnr > old_pwm:
                cols[7] = f" {new_psnr} "
                cols[8] = f" {new_ssim:.4f} "
                if old_pwm is not None and abs(new_psnr - old_pwm) > 0.1:
                    updated_psnr += 1

        # Apply done criteria
        ref_psnr = None
        pwm_psnr = None
        try:
            ref_psnr = float(cols[5].strip())
        except:
            pass
        try:
            pwm_psnr = float(cols[7].strip())
        except:
            pass

        should_be_done = False
        if "[proxy]" in algo_name:
            should_be_done = False
        elif ref_psnr is not None and pwm_psnr is not None:
            should_be_done = pwm_psnr >= ref_psnr - THRESHOLD
        # No ref = not done

        if should_be_done:
            cols[9] = " done "
            if old_status == "done":
                done_kept += 1
            else:
                done_added += 1
        else:
            cols[9] = "  "
            if old_status == "done":
                done_removed += 1

        lines[i] = "|".join(cols)

    STATE_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"Updated PWM PSNR for {updated_psnr} entries")
    print(f"Done status: {done_kept} kept, {done_added} added, {done_removed} removed")
    print(f"Total done: {done_kept + done_added}")
    print()

    # Show modalities with best improvements
    print("Top modality improvements (best solver vs old baseline):")
    improvements = []
    for mod, info in sorted(best_per_mod.items()):
        improvements.append((mod, info["psnr"], info["solver"]))
    for mod, psnr, solver in sorted(improvements, key=lambda x: -x[1])[:20]:
        print(f"  {mod:25s} best={psnr:6.1f} dB ({solver})")


if __name__ == "__main__":
    main()
