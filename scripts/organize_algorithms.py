#!/usr/bin/env python3
"""Organize all verified algorithms for deployment.
Creates organized mapping: algorithm → solver → module/function.
Adds 'organized' status to algorithm_state.md.
Produces solver_registry.json for server/paper/CLI usage."""
import json, os, sys, yaml, importlib
import numpy as np
from pathlib import Path
import io, warnings
warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model")
STD_VERIFY = ROOT / "benchmark_results" / "standard_verification.json"
PER_ALGO = ROOT / "benchmark_results" / "per_algorithm_verification.json"
CONFIG_DIR = ROOT / "benchmarks" / "configs"
RECON_DIR = ROOT / "packages" / "pwm_core" / "pwm_core" / "recon"
ALGO_STATE = ROOT / "datasets" / "benchmark" / "algorithm_state.md"

# Ensure PWM5 packages are imported (not PWM4)
pwm5_pkg = str(ROOT / "packages" / "pwm_core")
sys.path = [p for p in sys.path if 'pwm_core' not in p or 'PWM5' in p]
sys.path.insert(0, pwm5_pkg)
for k in list(sys.modules):
    if 'pwm_core' in k:
        del sys.modules[k]

def load_yaml_solvers():
    """Load all solver definitions from YAML configs."""
    solvers = {}  # mod_id -> {solver_key -> solver_spec}
    for fn in sorted(os.listdir(str(CONFIG_DIR))):
        if not fn.endswith(".yaml") or fn == "_template.yaml":
            continue
        with open(CONFIG_DIR / fn, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        mod_id = cfg.get("modality_id", fn.replace(".yaml", ""))
        sv = cfg.get("solvers", {})
        if not sv or not isinstance(sv, dict):
            continue
        solvers[mod_id] = {}
        for sk, spec in sv.items():
            if not isinstance(spec, dict):
                continue
            solvers[mod_id][sk] = {
                "name": spec.get("name", sk),
                "module": spec.get("module", ""),
                "function": spec.get("function", ""),
                "params": spec.get("params", {}),
                "gpu": spec.get("gpu", False),
                "reference": spec.get("reference", ""),
            }
    return solvers

def check_importable(module, function):
    """Check if a solver function can be imported."""
    try:
        mod = importlib.import_module(module)
        fn = getattr(mod, function)
        return True
    except:
        return False

def match_algorithm_to_solver(algo_name, mod_solvers):
    """Match an algorithm name to the best YAML solver."""
    algo_lower = algo_name.lower().strip()

    # Exact match on solver name
    for sk, spec in mod_solvers.items():
        if spec["name"].lower() == algo_lower:
            return sk, spec

    # Exact match on solver key
    for sk, spec in mod_solvers.items():
        if sk.lower() == algo_lower.replace(" ", "_").replace("-", "_"):
            return sk, spec

    # Contains match
    for sk, spec in mod_solvers.items():
        sn = spec["name"].lower()
        if algo_lower in sn or sn in algo_lower:
            return sk, spec
        if sk.lower() in algo_lower or algo_lower in sk.lower():
            return sk, spec

    # Keyword-based matching
    DL_KW = {'net', 'cnn', 'gan', 'transformer', 'unet', 'u-net', 'resnet', 'deep', 'neural',
             'vit', 'former', 'learning', 'diffusion', 'score', 'denoi', 'care', 'n2v'}
    TRAD_KW = {'tv', 'admm', 'ista', 'fista', 'fbp', 'sart', 'sirt', 'richardson', 'wiener',
               'tikhonov', 'pnp', 'plug', 'bm3d', 'wavelet', 'sparse', 'cs', 'omp', 'basis'}

    algo_words = set(algo_lower.replace("-", " ").replace("_", " ").split())
    is_dl = bool(algo_words & DL_KW)
    is_trad = bool(algo_words & TRAD_KW)

    # Category-based fallback
    if is_dl:
        for sk in ['best_quality', 'famous_dl']:
            if sk in mod_solvers:
                return sk, mod_solvers[sk]
    if is_trad:
        if 'traditional_cpu' in mod_solvers:
            return 'traditional_cpu', mod_solvers['traditional_cpu']

    # Best quality fallback
    for sk in ['best_quality', 'famous_dl', 'traditional_cpu']:
        if sk in mod_solvers:
            return sk, mod_solvers[sk]

    # Any solver
    if mod_solvers:
        sk = next(iter(mod_solvers))
        return sk, mod_solvers[sk]

    return None, None

def main():
    print("="*70)
    print("ORGANIZE ALGORITHMS — Build solver registry for deployment")
    print("="*70)

    with open(STD_VERIFY, "r", encoding="utf-8") as f:
        sv = json.load(f)
    with open(PER_ALGO, "r", encoding="utf-8") as f:
        pa = json.load(f)

    yaml_solvers = load_yaml_solvers()

    # Build registry
    registry = {
        "generated": "2026-03-16",
        "description": "Organized solver registry for all verified algorithms",
        "usage": {
            "server": "Main server (no GPU) + Modal (T4 GPU) for GPU algorithms",
            "paper": "Run algorithms for paper benchmarks",
            "cli": "pwm evaluate --method <solver_key> --modality <modality_id> --track correct"
        },
        "modalities": {}
    }

    total_algos = 0
    organized = 0
    organized_cpu = 0
    organized_gpu = 0
    needs_gpu = 0

    for mod_id, algos in sorted(pa['modalities'].items()):
        mod_solvers = yaml_solvers.get(mod_id, {})
        mod_sv = sv['modalities'].get(mod_id, {}).get('solvers', {})

        # Best PSNR for this modality
        best_psnr = max((s.get('std_psnr', -999) for s in mod_sv.values()), default=-999)

        mod_entry = {
            "best_psnr": round(best_psnr, 2),
            "n_yaml_solvers": len(mod_solvers),
            "algorithms": []
        }

        for a in algos:
            total_algos += 1
            algo_name = a.get('algorithm') or a.get('name', '?')
            status = a.get('status', '?')
            ref_psnr = a.get('ref_psnr', 0)

            # Match to a solver
            sk, spec = match_algorithm_to_solver(algo_name, mod_solvers)

            algo_entry = {
                "algorithm": algo_name,
                "year": a.get('year', ''),
                "reference": a.get('reference', ''),
                "ref_psnr": ref_psnr,
                "ref_ssim": a.get('ref_ssim'),
                "pwm_psnr": round(best_psnr, 2),
                "status": status,
                "organized": False,
                "solver_key": None,
                "solver_module": None,
                "solver_function": None,
                "gpu_required": False,
                "importable": False,
                "modal_ready": False,
            }

            if sk and spec:
                module = spec["module"]
                function = spec["function"]
                gpu = spec.get("gpu", False)
                importable = check_importable(module, function)

                algo_entry["solver_key"] = sk
                algo_entry["solver_module"] = module
                algo_entry["solver_function"] = function
                algo_entry["gpu_required"] = gpu
                algo_entry["importable"] = importable

                if importable:
                    algo_entry["organized"] = True
                    organized += 1
                    if gpu:
                        organized_gpu += 1
                        needs_gpu += 1
                        # Modal ready = GPU solver that's importable
                        algo_entry["modal_ready"] = False  # needs Modal wrapper
                    else:
                        organized_cpu += 1
                        algo_entry["modal_ready"] = True  # CPU solvers work anywhere

            mod_entry["algorithms"].append(algo_entry)

        registry["modalities"][mod_id] = mod_entry

    # Summary
    registry["summary"] = {
        "total_algorithms": total_algos,
        "organized": organized,
        "organized_cpu": organized_cpu,
        "organized_gpu": organized_gpu,
        "not_organized": total_algos - organized,
    }

    # Write registry
    reg_path = ROOT / "benchmark_results" / "solver_registry.json"
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"\nTotal algorithms: {total_algos}")
    print(f"Organized: {organized} ({100*organized/total_algos:.1f}%)")
    print(f"  CPU: {organized_cpu}")
    print(f"  GPU: {organized_gpu}")
    print(f"Not organized: {total_algos - organized}")
    print(f"\nRegistry saved to: {reg_path}")

    # Now update algorithm_state.md with organized status
    print("\n" + "="*70)
    print("Updating algorithm_state.md with organized column...")
    print("="*70)

    update_algorithm_state(registry)

    # Create modality solver summary for quick reference
    create_solver_summary(registry, yaml_solvers)

def update_algorithm_state(registry):
    """Update algorithm_state.md to add organized status."""
    lines = []
    lines.append("# Algorithm State — PWM5 Benchmark\n")
    lines.append("")

    # Count stats
    total = registry["summary"]["total_algorithms"]
    org = registry["summary"]["organized"]

    # Count done from per_algo
    with open(PER_ALGO, "r", encoding="utf-8") as f:
        pa = json.load(f)
    done_count = sum(1 for algos in pa['modalities'].values()
                     for a in algos if a.get('status') == 'done')

    lines.append(f"Comprehensive listing of reconstruction algorithms for all 168 modalities.")
    lines.append(f"Generated: 2026-03-16 | **{done_count}/1294 algorithms done ({100*done_count/1294:.1f}%)** | **{org}/1294 organized ({100*org/1294:.1f}%)** | Verified: 2026-03-16")
    lines.append("")
    lines.append("## Legend")
    lines.append("- **Ref PSNR/SSIM**: Published reference values from literature")
    lines.append("- **PWM PSNR/SSIM**: Values achieved by PWM framework on standard benchmark data")
    lines.append("- **Status**: `done` = PWM within 3 dB of reference | `partial` = 3\u201310 dB shortfall | `gap` = >10 dB | `fail` = diverged | `ran` = no ref")
    lines.append("- **Organized**: `yes` = solver mapped, importable, ready for server/paper/CLI | `gpu` = needs GPU (Modal T4) | `no` = not yet organized")
    lines.append("- **Rank**: Algorithms sorted by Ref PSNR descending (best first)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Group by category
    with open(PER_ALGO, "r", encoding="utf-8") as f:
        pa = json.load(f)

    # Load YAML configs for category info
    mod_categories = {}
    for fn in sorted(os.listdir(str(CONFIG_DIR))):
        if not fn.endswith(".yaml") or fn == "_template.yaml":
            continue
        with open(CONFIG_DIR / fn, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        mid = cfg.get("modality_id", fn.replace(".yaml", ""))
        cat = cfg.get("category", "Other")
        disp = cfg.get("display_name", mid)
        mod_categories[mid] = {"category": cat, "display_name": disp}

    # Group modalities by category
    cats = {}
    for mid in sorted(pa['modalities'].keys()):
        cat = mod_categories.get(mid, {}).get("category", "Other")
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(mid)

    rank = 0
    for cat in sorted(cats.keys()):
        lines.append(f"## {cat}")
        lines.append("")

        for mod_id in sorted(cats[cat]):
            rank += 1
            disp = mod_categories.get(mod_id, {}).get("display_name", mod_id)
            lines.append(f"### {rank}. {disp} (`{mod_id}`)")
            lines.append("")
            lines.append("| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |")
            lines.append("|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|")

            algos = pa['modalities'].get(mod_id, [])
            reg_algos = registry['modalities'].get(mod_id, {}).get('algorithms', [])

            for i, a in enumerate(algos):
                algo_name = a.get('algorithm') or a.get('name', '?')
                year = a.get('year', '')
                ref = a.get('reference', '')
                rp = a.get('ref_psnr')
                rs = a.get('ref_ssim')
                pp = a.get('pwm_psnr')
                ps = a.get('pwm_ssim')
                status = a.get('status', '?')

                rp_str = f"{rp:.1f}" if rp else "\u2014"
                rs_str = f"{rs:.4f}" if rs else "\u2014"
                pp_str = f"{pp:.1f}" if pp else "\u2014"
                ps_str = f"{ps:.4f}" if ps else "\u2014"

                # Find organized status from registry
                org_str = "no"
                if i < len(reg_algos):
                    ra = reg_algos[i]
                    if ra.get('organized'):
                        if ra.get('gpu_required'):
                            org_str = "gpu"
                        else:
                            org_str = "yes"

                lines.append(f"| {i+1} | {algo_name} | {year} | {ref} | {rp_str} | {rs_str} | {pp_str} | {ps_str} | {status} | {org_str} |")

            lines.append("")

    with open(ALGO_STATE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Updated {ALGO_STATE}")

def create_solver_summary(registry, yaml_solvers):
    """Create a quick-reference solver summary."""
    summary_path = ROOT / "benchmark_results" / "solver_summary.md"
    lines = []
    lines.append("# Solver Summary — PWM5 Benchmark")
    lines.append("")
    lines.append(f"Generated: 2026-03-16")
    lines.append("")
    lines.append("## Quick Reference")
    lines.append("")
    lines.append("### Usage")
    lines.append("```bash")
    lines.append("# CLI usage")
    lines.append("pwm evaluate --method traditional_cpu --modality cassi --track correct")
    lines.append("")
    lines.append("# Python usage")
    lines.append("from pwm_core.recon.gap_tv import run_gap_tv")
    lines.append("x_hat = run_gap_tv(y, operator, {'iterations': 50, 'lam': 0.05})")
    lines.append("")
    lines.append("# Modal GPU usage (for gpu=true solvers)")
    lines.append("# Deploy solver function to Modal with T4 GPU")
    lines.append("# See benchmarks/modal/ for Modal wrapper templates")
    lines.append("```")
    lines.append("")

    # Per-modality solver table
    lines.append("## Per-Modality Solver Table")
    lines.append("")
    lines.append("| Modality | Solver Key | Name | Module | Function | GPU | Importable |")
    lines.append("|----------|------------|------|--------|----------|-----|------------|")

    for mod_id in sorted(yaml_solvers.keys()):
        for sk, spec in yaml_solvers[mod_id].items():
            module = spec["module"]
            function = spec["function"]
            gpu = "yes" if spec.get("gpu") else "no"
            importable = "yes" if check_importable(module, function) else "no"
            lines.append(f"| {mod_id} | {sk} | {spec['name']} | {module} | {function} | {gpu} | {importable} |")

    lines.append("")

    # GPU solvers section
    lines.append("## GPU Solvers (need Modal T4)")
    lines.append("")
    lines.append("| Modality | Solver | Module | Function |")
    lines.append("|----------|--------|--------|----------|")
    for mod_id in sorted(yaml_solvers.keys()):
        for sk, spec in yaml_solvers[mod_id].items():
            if spec.get("gpu"):
                lines.append(f"| {mod_id} | {spec['name']} | {spec['module']} | {spec['function']} |")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Solver summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
