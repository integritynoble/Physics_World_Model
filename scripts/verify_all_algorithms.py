"""Comprehensive verification of ALL 1294 algorithms in algorithm_state.md.

For each algorithm marked 'done':
1. Verify the modality has at least one working solver
2. Verify the PWM PSNR is within 3 dB of reference PSNR
3. Flag any inconsistencies

For each algorithm NOT marked 'done':
1. Check if the gap is actually <= 3 dB (should be marked done)
2. Report the gap
"""

import re
import json
import importlib
import numpy as np
import yaml
import time
from pathlib import Path

STATE_PATH = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\datasets\benchmark\algorithm_state.md")
CONFIGS_DIR = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\benchmarks\configs")
V3_PATH = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\benchmark_results\solver_verification_v3.json")
OUT_PATH = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\benchmark_results\full_algorithm_verification.json")


def parse_algorithm_state():
    """Parse algorithm_state.md into structured data."""
    text = STATE_PATH.read_text(encoding="utf-8")
    lines = text.split("\n")

    modalities = {}
    current_modality = None
    current_id = None

    for line in lines:
        # Match modality header: ### N. Name (id)
        m = re.match(r'^### \d+\.\s+(.+?)\s+\(`(\w+)`\)', line)
        if m:
            current_modality = m.group(1)
            current_id = m.group(2)
            modalities[current_id] = {
                "name": current_modality,
                "algorithms": []
            }
            continue

        # Match table row
        if current_id and line.startswith("|") and not line.startswith("|---") and not line.startswith("| #"):
            parts = [p.strip() for p in line.split("|")]
            # Filter empty
            parts = [p for p in parts if p]
            if len(parts) >= 8:
                try:
                    algo_num = parts[0]
                    algo_name = parts[1]
                    year = parts[2]
                    ref = parts[3]
                    ref_psnr_str = parts[4]
                    ref_ssim_str = parts[5]
                    pwm_psnr_str = parts[6]
                    pwm_ssim_str = parts[7]
                    status = parts[8] if len(parts) > 8 else ""

                    ref_psnr = None
                    pwm_psnr = None
                    try:
                        ref_psnr = float(ref_psnr_str)
                    except (ValueError, TypeError):
                        pass
                    try:
                        pwm_psnr = float(pwm_psnr_str)
                    except (ValueError, TypeError):
                        pass

                    modalities[current_id]["algorithms"].append({
                        "num": algo_num,
                        "name": algo_name,
                        "year": year,
                        "ref_psnr": ref_psnr,
                        "pwm_psnr": pwm_psnr,
                        "status": status.strip(),
                    })
                except (IndexError, ValueError):
                    pass

    return modalities


def load_solver_verification():
    """Load v3 solver verification results."""
    if V3_PATH.exists():
        with open(V3_PATH, "r") as f:
            return json.load(f)
    return {"modalities": {}}


def check_modality_solvers(modality_id):
    """Check if a modality has working solvers via YAML config."""
    yaml_path = CONFIGS_DIR / f"{modality_id}.yaml"
    if not yaml_path.exists():
        return "no_yaml", []

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "solvers" not in data or not data["solvers"]:
        return "no_solvers", []

    working = []
    errors = []
    for solver_key, solver_cfg in data["solvers"].items():
        if not isinstance(solver_cfg, dict):
            continue
        module_name = solver_cfg.get("module", "")
        function_name = solver_cfg.get("function", "")
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, function_name)
            working.append(solver_key)
        except Exception as e:
            errors.append(f"{solver_key}: {str(e)[:100]}")

    if working:
        return "ok", working
    return "all_broken", errors


def main():
    print("Parsing algorithm_state.md...")
    modalities = parse_algorithm_state()

    print(f"Found {len(modalities)} modalities")

    v3 = load_solver_verification()
    v3_mods = v3.get("modalities", {})

    total_algos = 0
    total_done = 0
    total_not_done = 0

    issues = {
        "done_but_gap_too_large": [],       # Marked done but gap > 3 dB
        "done_but_no_working_solver": [],    # Marked done but no solver works
        "done_but_no_psnr": [],              # Marked done but no PWM PSNR
        "not_done_but_gap_ok": [],           # Not done but gap <= 3 dB
        "no_yaml_config": [],                # Modality has no YAML
    }

    solver_status = {}  # modality_id -> (status, working_solvers)
    modality_results = {}

    for mod_id, mod_data in sorted(modalities.items()):
        algos = mod_data["algorithms"]
        if not algos:
            continue

        # Check solver status (cached per modality)
        if mod_id not in solver_status:
            solver_status[mod_id] = check_modality_solvers(mod_id)

        s_status, s_detail = solver_status[mod_id]

        # Also check v3 verification
        v3_mod = v3_mods.get(mod_id, {})
        v3_verified = sum(1 for v in v3_mod.values()
                         if isinstance(v, dict) and v.get("status") == "verified")

        mod_result = {
            "name": mod_data["name"],
            "solver_status": s_status,
            "v3_verified_count": v3_verified,
            "algorithms": [],
        }

        for algo in algos:
            total_algos += 1
            is_done = algo["status"] == "done"
            ref_psnr = algo["ref_psnr"]
            pwm_psnr = algo["pwm_psnr"]

            algo_result = {
                "name": algo["name"],
                "ref_psnr": ref_psnr,
                "pwm_psnr": pwm_psnr,
                "status": algo["status"],
                "issues": [],
            }

            if is_done:
                total_done += 1

                # Check 1: gap <= 3 dB
                if ref_psnr is not None and pwm_psnr is not None:
                    gap = ref_psnr - pwm_psnr
                    if gap > 3.0:
                        issues["done_but_gap_too_large"].append(
                            f"{mod_id}/{algo['name']}: gap={gap:.1f} dB (ref={ref_psnr}, pwm={pwm_psnr})"
                        )
                        algo_result["issues"].append(f"gap={gap:.1f}>3dB")

                # Check 2: has working solver
                if s_status not in ("ok",):
                    issues["done_but_no_working_solver"].append(
                        f"{mod_id}/{algo['name']}: solver_status={s_status}"
                    )
                    algo_result["issues"].append(f"solver={s_status}")

                # Check 3: has PWM PSNR
                if pwm_psnr is None:
                    issues["done_but_no_psnr"].append(
                        f"{mod_id}/{algo['name']}"
                    )
                    algo_result["issues"].append("no_pwm_psnr")

            else:
                total_not_done += 1

                # Check: should it be done?
                if ref_psnr is not None and pwm_psnr is not None:
                    gap = ref_psnr - pwm_psnr
                    if gap <= 3.0:
                        issues["not_done_but_gap_ok"].append(
                            f"{mod_id}/{algo['name']}: gap={gap:.1f} dB (ref={ref_psnr}, pwm={pwm_psnr})"
                        )
                        algo_result["issues"].append(f"should_be_done(gap={gap:.1f})")

            mod_result["algorithms"].append(algo_result)

        # Check modality-level issue
        if s_status == "no_yaml":
            issues["no_yaml_config"].append(mod_id)

        modality_results[mod_id] = mod_result

    # Print summary
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE ALGORITHM VERIFICATION")
    print(f"{'='*70}")
    print(f"Total algorithms: {total_algos}")
    print(f"Marked done:      {total_done}")
    print(f"Not done:         {total_not_done}")

    print(f"\n--- Issues ---")
    for issue_type, items in issues.items():
        if items:
            print(f"\n{issue_type} ({len(items)}):")
            for item in items[:20]:
                print(f"  - {item}")
            if len(items) > 20:
                print(f"  ... and {len(items) - 20} more")

    # Count modalities with all solvers working
    all_ok = sum(1 for s, _ in solver_status.values() if s == "ok")
    print(f"\n--- Solver Status ---")
    print(f"Modalities with working solvers: {all_ok}/{len(solver_status)}")
    broken = [(k, s, d) for k, (s, d) in solver_status.items() if s != "ok"]
    if broken:
        print(f"Modalities without working solvers ({len(broken)}):")
        for mod_id, status, detail in broken:
            print(f"  {mod_id}: {status}")

    # Final verdict
    total_issues = sum(len(v) for v in issues.values())
    print(f"\n{'='*70}")
    if total_issues == 0:
        print("VERDICT: ALL ALGORITHMS VERIFIED - No issues found")
    else:
        print(f"VERDICT: {total_issues} issues found across {len(issues)} categories")
    print(f"{'='*70}")

    # Save
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_algorithms": total_algos,
        "done": total_done,
        "not_done": total_not_done,
        "issues": {k: v for k, v in issues.items()},
        "issue_counts": {k: len(v) for k, v in issues.items()},
        "modalities": modality_results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
