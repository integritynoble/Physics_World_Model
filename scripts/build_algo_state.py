#!/usr/bin/env python3
"""Build comprehensive per-algorithm state.md for all modalities.

For each modality, lists:
- Every solver actually tested (from JSON results) with PSNR/SSIM
- Every YAML-defined target algorithm with implementation status
- Reference leaderboard entries from https://pwm.platformai.org/benchmark
"""
import yaml
import json
import sys
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))

CONFIG_DIR = ROOT / "benchmarks" / "configs"
RESULTS_PATH = ROOT / "benchmark_results" / "comprehensive_algorithm_test.json"
LEADERBOARD_PATH = ROOT / "benchmark_results" / "benchmark_leaderboard_reference.json"
STATE_PATH = ROOT / "datasets" / "benchmark" / "state.md"

# Load existing test results
with open(RESULTS_PATH) as f:
    results = json.load(f)

# Load leaderboard reference data
with open(LEADERBOARD_PATH) as f:
    leaderboard_ref = json.load(f)

# Human-readable names for generic JSON solver keys
GENERIC_KEY_NAMES = {
    "precomputed_baseline": "Precomputed Baseline",
    "precomputed_recon": "Precomputed Reconstruction",
    "precomputed_wiener": "Wiener Filter (precomputed)",
    "precomputed_fbp": "FBP (precomputed)",
    "precomputed_phase_baseline": "Phase Baseline (precomputed)",
    "fbp_ramlak": "FBP (Ram-Lak filter)",
    "fbp_shepp_logan": "FBP (Shepp-Logan filter)",
    "sart_10iter": "SART (10 iterations)",
    "zero_filled": "Zero-Filled (IFFT)",
    "cs_mri_wavelet": "CS-MRI (Wavelet L1)",
    "sense": "SENSE (parallel imaging)",
    "rl_20iter": "Richardson-Lucy (20 iter)",
    "rl_50iter": "Richardson-Lucy (50 iter)",
    "rl_ctf_20iter": "Richardson-Lucy + CTF (20 iter)",
    "gap_tv": "GAP-TV",
    "mask_division_baseline": "Mask Division Baseline",
    "wiener_deconv": "Wiener Deconvolution",
    "wiener_sim": "Wiener SIM Reconstruction",
    "fourier_notch": "Fourier Notch Filter",
    "bscan_baseline": "B-scan Direct (noisy)",
    "bscan_ideal_baseline": "B-scan Ideal (noiseless)",
    "sqrt_intensity_amplitude": "sqrt(Intensity) Amplitude",
    "wrapped_phase_baseline": "Wrapped Phase Baseline",
    "bicubic_upsample": "Bicubic Upsampling",
    "direct_render_baseline": "Direct Render Baseline",
    "spectral_shift_baseline": "Spectral Shift Baseline",
    "backprojection_baseline": "Backprojection Baseline",
}


def check_importable(module, fn):
    try:
        m = importlib.import_module(module)
        return getattr(m, fn, None) is not None
    except Exception:
        return False


# Load all modality configs
all_mods = {}
for f in sorted(CONFIG_DIR.glob("*.yaml")):
    if f.name.startswith("_"):
        continue
    with open(f, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    mod_id = data.get("modality_id", f.stem)
    all_mods[mod_id] = data

# Group by category
categories = {}
for mod_id, data in all_mods.items():
    cat = data.get("category", "Other")
    categories.setdefault(cat, []).append(mod_id)

# Build state.md content
lines = []
lines.append("# Benchmark Algorithm Test State")
lines.append("")
lines.append("Last updated: 2026-03-11 — 166 modalities with datasets, 168 with YAML configs")
lines.append("")
lines.append("## Legend")
lines.append("- `done (X.XX dB)`: tested, PSNR/SSIM recorded in benchmark run")
lines.append("- `importable`: module implemented in pwm_core, ready to run")
lines.append("- `pending`: module not yet implemented")
lines.append("- `reference`: algorithm from https://pwm.platformai.org/benchmark leaderboard")
lines.append("")
lines.append("Reference leaderboard: https://pwm.platformai.org/benchmark")
lines.append(f"Total algorithms on leaderboard: 1,367 across 168 modalities")
lines.append("")

# Count stats
summary_counts = {"dataset_done": 0, "solvers_tested": 0, "modalities_any_test": 0}

# Quick status table
lines.append("## Quick Status Table")
lines.append("")
lines.append("| Modality | Dataset | Tested | Best PSNR | YAML Algos | Leaderboard Ref | Speclab |")
lines.append("|----------|---------|--------|-----------|------------|-----------------|---------|")

for mod_id in sorted(all_mods.keys()):
    data = all_mods[mod_id]
    solvers = data.get("solvers", {}) or {}

    # Dataset status
    bench_dir = ROOT / "datasets" / "benchmark" / mod_id
    pub_dir = bench_dir / "public"
    has_h5 = bool(list(pub_dir.glob("*.h5"))) if pub_dir.exists() else \
              bool(list(bench_dir.glob("*_public*.h5"))) if bench_dir.exists() else False
    if mod_id == "ct":
        has_h5 = (bench_dir / "public").exists()
    dataset_status = "done" if has_h5 else "pending"
    if has_h5:
        summary_counts["dataset_done"] += 1

    # JSON test results
    mod_results = results.get("modalities", {}).get(mod_id, {})
    tested_solvers = {k: v for k, v in mod_results.get("solvers", {}).items()
                      if v.get("psnr_db") is not None}
    n_tested = len(tested_solvers)
    summary_counts["solvers_tested"] += n_tested
    if n_tested > 0:
        summary_counts["modalities_any_test"] += 1

    best_psnr = max((v["psnr_db"] for v in tested_solvers.values()), default=None)
    best_psnr_str = f"{best_psnr:.1f} dB" if best_psnr is not None else "—"

    n_yaml = len([k for k, v in solvers.items() if v])
    n_ref = len(leaderboard_ref.get(mod_id, []))
    ref_str = f"{n_ref}" if n_ref > 0 else "—"

    lines.append(f"| {mod_id} | {dataset_status} | {n_tested} | {best_psnr_str} | {n_yaml} | {ref_str} | pending |")

lines.append("")
lines.append(f"**Summary:** {summary_counts['dataset_done']}/168 datasets done, "
             f"{summary_counts['solvers_tested']} solver tests recorded across "
             f"{summary_counts['modalities_any_test']}/166 modalities with datasets")
lines.append("")

# Detailed per-modality section
lines.append("---")
lines.append("")
lines.append("## Detailed Algorithm Test Results by Modality")
lines.append("")
lines.append("Each modality shows three sections:")
lines.append("1. **Tested** — results from our benchmark runs (PSNR/SSIM recorded)")
lines.append("2. **YAML Solvers** — algorithms defined in config (implementation status)")
lines.append("3. **Leaderboard Reference** — top algorithms from pwm.platformai.org/benchmark")
lines.append("")

for cat, mod_ids in sorted(categories.items()):
    lines.append(f"### {cat}")
    lines.append("")

    for mod_id in sorted(mod_ids):
        data = all_mods[mod_id]
        solvers = data.get("solvers", {}) or {}
        display = data.get("display_name", mod_id)
        ref_algos = leaderboard_ref.get(mod_id, [])

        lines.append(f"#### {mod_id} — {display}")
        lines.append("")
        lines.append("| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |")
        lines.append("|---|-----------|------|--------|-----------|------|---------|--------|")

        row_num = 1

        # Section 1: Actually tested solvers (from JSON)
        mod_results = results.get("modalities", {}).get(mod_id, {})
        tested = mod_results.get("solvers", {})

        for sk, sr in sorted(tested.items()):
            if not sr:
                continue
            psnr = sr.get("psnr_db")
            ssim = sr.get("ssim")
            t = sr.get("exec_time_sec")
            name = GENERIC_KEY_NAMES.get(sk, sk.replace("_", " ").title())

            if psnr is not None:
                status_str = "**done**"
                psnr_str = f"{psnr:.2f}"
                ssim_str = f"{ssim:.4f}" if ssim is not None else "—"
                t_str = f"{t:.2f}" if t is not None else "0.00"
                algo_type = "traditional"
            else:
                status_str = "no result"
                psnr_str = ssim_str = t_str = "—"
                algo_type = "—"

            lines.append(f"| {row_num} | {name} | {algo_type} | {status_str} | {psnr_str} | {ssim_str} | {t_str} | benchmark run |")
            row_num += 1

        # Section 2: YAML-defined solvers not already tested
        tested_keys = set(tested.keys())
        for sk, sv in solvers.items():
            if not sv:
                continue
            name = sv.get("name", "?")
            module = sv.get("module", "")
            fn = sv.get("function", "")

            # Check if already shown
            if sk in tested_keys:
                continue

            # Determine type from key
            if "dl" in sk or sk in ("best_quality", "small_gpu"):
                algo_type = "deep_learning"
            else:
                algo_type = "traditional"

            if check_importable(module, fn):
                status_str = "importable"
            else:
                status_str = "pending"

            mod_fn = f"{module.split('.')[-1]}.{fn}" if module else "—"
            lines.append(f"| {row_num} | {name} | {algo_type} | {status_str} | — | — | — | YAML config |")
            row_num += 1

        # Section 3: Leaderboard reference algorithms
        if ref_algos:
            for entry in ref_algos:
                r_name = entry.get("name", "?")
                r_type = entry.get("type", "—")
                r_psnr = entry.get("psnr_db")
                r_ssim = entry.get("ssim")
                r_rank = entry.get("rank", "?")

                psnr_str = f"{r_psnr:.1f}" if r_psnr is not None else "—"
                ssim_str = f"{r_ssim:.3f}" if r_ssim is not None else "—"

                lines.append(f"| #{r_rank} | {r_name} | {r_type} | reference | {psnr_str} | {ssim_str} | — | pwm.platformai.org |")

        lines.append("")

# Write
content = "\n".join(lines)
STATE_PATH.write_text(content, encoding="utf-8")
print(f"Written: {STATE_PATH}")
print(f"Lines: {len(lines)}")
print(f"Datasets done: {summary_counts['dataset_done']}/168")
print(f"Solver tests recorded: {summary_counts['solvers_tested']}")
print(f"Modalities with any test: {summary_counts['modalities_any_test']}")
modalities_with_ref = len([m for m in all_mods if leaderboard_ref.get(m)])
print(f"Modalities with leaderboard reference: {modalities_with_ref}")
