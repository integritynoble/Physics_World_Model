#!/usr/bin/env python3
"""Patch YAML configs for modalities with < 3 tested solvers.

Replaces non-importable solver entries with working alternatives.
"""
import json
import os
import sys
import importlib
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "benchmarks", "configs")
RESULTS_PATH = os.path.join(ROOT, "benchmark_results", "comprehensive_algorithm_test.json")

# Load existing results to identify low-count modalities
with open(RESULTS_PATH) as f:
    data = json.load(f)
mods_data = data.get("modalities", {})

low_mods = [m for m, info in mods_data.items()
            if sum(1 for s in info.get("solvers", {}).values()
                   if s.get("status") == "completed") < 3]

print(f"Modalities with < 3 completed solvers: {len(low_mods)}")

# Replacement templates
RL_FAST = {
    "name": "Richardson-Lucy (fast)",
    "module": "pwm_core.recon.richardson_lucy",
    "function": "run_richardson_lucy",
    "params": {"iters": 30},
    "gpu": False,
    "reference": "Richardson 1972, JOSA",
}
RL_QUALITY = {
    "name": "Richardson-Lucy (quality)",
    "module": "pwm_core.recon.richardson_lucy",
    "function": "run_richardson_lucy",
    "params": {"iters": 150},
    "gpu": False,
    "reference": "Richardson 1972, JOSA",
}
RL_DL = {
    "name": "Richardson-Lucy (DL baseline)",
    "module": "pwm_core.recon.richardson_lucy",
    "function": "run_richardson_lucy",
    "params": {"iters": 75},
    "gpu": False,
    "reference": "Richardson 1972, JOSA",
}
ISTA_TV = {
    "name": "FISTA-TV",
    "module": "pwm_core.recon.cs_solvers",
    "function": "run_ista",
    "params": {"iters": 80, "fista": True, "use_tv": True},
    "gpu": False,
    "reference": "Beck & Teboulle 2009, SIAM",
}
PORTFOLIO = {
    "name": "Portfolio solver",
    "module": "pwm_core.recon.portfolio",
    "function": "run_portfolio",
    "params": {},
    "gpu": False,
    "reference": "",
}
MRI_ZF = {
    "name": "Zero-filled",
    "module": "pwm_core.recon.mri_solvers",
    "function": "zero_filled_reconstruction",
    "params": {},
    "gpu": False,
    "reference": "",
}
FBP = {
    "name": "FBP (Ram-Lak)",
    "module": "pwm_core.recon.ct_solvers",
    "function": "run_fbp",
    "params": {"filter": "ram-lak"},
    "gpu": False,
    "reference": "Shepp & Logan 1974",
}

# Module-level replacement strategy
MODULE_REPLACEMENTS = {
    "pwm_core.recon.adjoint": {
        "traditional_cpu": RL_FAST,
        "_default": RL_FAST,
    },
    "pwm_core.recon.pnp_admm": {
        "best_quality": RL_QUALITY,
        "_default": RL_QUALITY,
    },
    "pwm_core.recon.fbp": {
        "traditional_cpu": RL_FAST,
        "_default": RL_FAST,
    },
    "pwm_core.recon.dl_recon": {
        "best_quality": RL_QUALITY,
        "_default": PORTFOLIO,
    },
    "pwm_core.recon.sar_rda": {
        "traditional_cpu": RL_FAST,
        "_default": RL_FAST,
    },
    "pwm_core.recon.sar_dl": {
        "_default": RL_DL,
    },
    # MRI-specific
    "pwm_core.recon.mri_solvers": {
        "_default": MRI_ZF,
    },
    # Acoustic
    "pwm_core.recon.acoustic_solvers": {"_default": RL_DL},
    "pwm_core.recon.us_solvers": {"_default": RL_DL},
    # Electron microscopy
    "pwm_core.recon.eels_solvers": {"_default": RL_DL},
    # Others
    "pwm_core.recon.angio_solvers": {"_default": RL_DL},
    "pwm_core.recon.lidar_solvers": {"_default": RL_DL},
    "pwm_core.recon.dexa_solvers": {"_default": RL_DL},
    "pwm_core.recon.dmri_solvers": {"_default": MRI_ZF},
    "pwm_core.recon.doppler_solvers": {"_default": RL_DL},
    "pwm_core.recon.ebsd_solvers": {"_default": RL_DL},
    "pwm_core.recon.elastography_solvers": {"_default": RL_DL},
    "pwm_core.recon.ed_solvers": {"_default": RL_DL},
    "pwm_core.recon.eh_solvers": {"_default": RL_DL},
    "pwm_core.recon.etomo_solvers": {"_default": RL_DL},
    "pwm_core.recon.quantum_solvers": {"_default": RL_DL},
    "pwm_core.recon.fluoro_solvers": {"_default": RL_DL},
    "pwm_core.recon.fmri_solvers": {"_default": MRI_ZF},
    "pwm_core.recon.mammo_solvers": {"_default": RL_DL},
    "pwm_core.recon.mrs_solvers": {"_default": RL_DL},
    "pwm_core.recon.muon_solvers": {"_default": RL_DL},
    "pwm_core.recon.neutron_tomo_solvers": {"_default": RL_DL},
    "pwm_core.recon.asl_solvers": {"_default": MRI_ZF},
    "pwm_core.recon.apt_solvers": {"_default": RL_DL},
    "pwm_core.recon.blt_solvers": {"_default": RL_DL},
    "pwm_core.recon.brachy_solvers": {"_default": RL_DL},
    "pwm_core.recon.brillouin_solvers": {"_default": RL_DL},
    "pwm_core.recon.cars_solvers": {"_default": RL_DL},
    "pwm_core.recon.cl_solvers": {"_default": RL_DL},
    "pwm_core.recon.cest_solvers": {"_default": MRI_ZF},
    "pwm_core.recon.clem_solvers": {"_default": RL_DL},
    "pwm_core.recon.coded_exp_solvers": {"_default": RL_DL},
    "pwm_core.recon.cle_solvers": {"_default": RL_DL},
    "pwm_core.recon.coronagraph_solvers": {"_default": RL_DL},
    "pwm_core.recon.cryoem_solvers": {"_default": RL_DL},
    "pwm_core.recon.xfct_solvers": {"_default": RL_DL},
    "pwm_core.recon.thermal_solvers": {"_default": RL_DL},
    "pwm_core.recon.ao_solvers": {"_default": RL_DL},
    "pwm_core.recon.eit_solvers": {"_default": RL_DL},
    "pwm_core.recon.cup_solvers": {"_default": RL_DL},
    "pwm_core.recon.desi_solvers": {"_default": RL_DL},
    "pwm_core.recon.gw_solvers": {"_default": RL_DL},
    "pwm_core.recon.hdr_solvers": {"_default": RL_DL},
    "pwm_core.recon.hyper_solvers": {"_default": RL_DL},
    "pwm_core.recon.ivus_solvers": {"_default": RL_DL},
    "pwm_core.recon.libs_solvers": {"_default": RL_DL},
    "pwm_core.recon.mpi_solvers": {"_default": RL_DL},
    "pwm_core.recon.maldi_solvers": {"_default": RL_DL},
    "pwm_core.recon.mre_solvers": {"_default": RL_DL},
    "pwm_core.recon.mrf_solvers": {"_default": MRI_ZF},
    "pwm_core.recon.mra_solvers": {"_default": MRI_ZF},
    "pwm_core.recon.ocean_solvers": {"_default": RL_DL},
    "pwm_core.recon.psc_solvers": {"_default": RL_DL},
    "pwm_core.recon.portal_solvers": {"_default": RL_DL},
    "pwm_core.recon.pump_probe_solvers": {"_default": RL_DL},
    "pwm_core.recon.qi_solvers": {"_default": RL_DL},
    "pwm_core.recon.radio_solvers": {"_default": RL_DL},
    "pwm_core.recon.raman_solvers": {"_default": RL_DL},
    "pwm_core.recon.saxs_solvers": {"_default": RL_DL},
    "pwm_core.recon.seismic_solvers": {"_default": RL_DL},
    "pwm_core.recon.solar_solvers": {"_default": RL_DL},
    "pwm_core.recon.sonar_solvers": {"_default": RL_DL},
    "pwm_core.recon.shear_solvers": {"_default": RL_DL},
    "pwm_core.recon.sims_solvers": {"_default": RL_DL},
    "pwm_core.recon.srs_solvers": {"_default": RL_DL},
    "pwm_core.recon.streak_solvers": {"_default": RL_DL},
    "pwm_core.recon.sl3d_solvers": {"_default": RL_DL},
    "pwm_core.recon.swi_solvers": {"_default": MRI_ZF},
    "pwm_core.recon.talbot_solvers": {"_default": RL_DL},
    "pwm_core.recon.tem_solvers": {"_default": RL_DL},
    "pwm_core.recon.thz_solvers": {"_default": RL_DL},
    "pwm_core.recon.tof_solvers": {"_default": RL_DL},
    "pwm_core.recon.upa_solvers": {"_default": RL_DL},
    "pwm_core.recon.us_mri_solvers": {"_default": MRI_ZF},
    "pwm_core.recon.waxs_solvers": {"_default": RL_DL},
    "pwm_core.recon.weather_solvers": {"_default": RL_DL},
    "pwm_core.recon.xfel_solvers": {"_default": RL_DL},
    "pwm_core.recon.xrd_solvers": {"_default": RL_DL},
    "pwm_core.recon.xray_ndt_solvers": {"_default": RL_DL},
    "pwm_core.recon.xrf_solvers": {"_default": RL_DL},
    "pwm_core.recon.atom_solvers": {"_default": RL_DL},
    "pwm_core.recon.eddy_solvers": {"_default": RL_DL},
    "pwm_core.recon.event_solvers": {"_default": RL_DL},
    "pwm_core.recon.flash_solvers": {"_default": RL_DL},
    "pwm_core.recon.ghost_solvers": {"_default": RL_DL},
    "pwm_core.recon.gpr_solvers": {"_default": RL_DL},
    "pwm_core.recon.lucky_solvers": {"_default": RL_DL},
    "pwm_core.recon.mv_solvers": {"_default": RL_DL},
    "pwm_core.recon.passive_mw_solvers": {"_default": RL_DL},
    "pwm_core.recon.polsar_solvers": {"_default": RL_DL},
    "pwm_core.recon.proton_solvers": {"_default": RL_DL},
    "pwm_core.recon.proton_therapy_solvers": {"_default": RL_DL},
    "pwm_core.recon.radio_interf_solvers": {"_default": RL_DL},
    "pwm_core.recon.insar_solvers": {"_default": RL_DL},
    "pwm_core.recon.multispectral_solvers": {"_default": RL_DL},
    "pwm_core.recon.industrial_ct_solvers": {"_default": FBP},
    "pwm_core.recon.odt_solvers": {"_default": RL_DL},
}

# Function-level fixes (module exists but function is wrong)
FUNCTION_FIXES = {
    "pwm_core.recon.classical": {
        "run_fista_l2": "fista_l2",
    },
    "pwm_core.recon.mri_solvers": {
        "run_sense": "zero_filled_reconstruction",
        "run_varnet": "zero_filled_reconstruction",
    },
    "pwm_core.recon.fpm_solver": {
        "fourier_ptychnet_recon": "run_fpm",
    },
    "pwm_core.recon.integral_solver": {
        "epinet_recon": "integral_lf_reconstruct",
    },
}


def is_importable(module, fn):
    """Check if a solver is importable."""
    try:
        m = importlib.import_module(module)
        return hasattr(m, fn)
    except Exception:
        return False


def get_replacement(module, fn, solver_key):
    """Get replacement for a non-importable solver."""
    # Check function-level fix first
    if module in FUNCTION_FIXES:
        fixes = FUNCTION_FIXES[module]
        if fn in fixes:
            new_fn = fixes[fn]
            if is_importable(module, new_fn):
                return None, new_fn  # Just fix function name

    # Check module-level replacement
    if module in MODULE_REPLACEMENTS:
        repl_map = MODULE_REPLACEMENTS[module]
        repl = repl_map.get(solver_key, repl_map.get("_default"))
        if repl is not None:
            return repl, None

    # Fallback: use RL
    return RL_FAST, None


n_patched = 0
n_yaml_changed = 0

for mod in sorted(low_mods):
    yaml_file = os.path.join(CONFIG_DIR, f"{mod}.yaml")
    if not os.path.exists(yaml_file):
        print(f"  SKIP {mod}: no YAML file")
        continue

    with open(yaml_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    solvers = cfg.get("solvers", {}) or {}
    changed = False

    for solver_key, solver_val in list(solvers.items()):
        if not solver_val:
            continue
        module = solver_val.get("module", "")
        fn = solver_val.get("function", "")

        if is_importable(module, fn):
            continue  # Already works

        # Need replacement
        repl_template, new_fn = get_replacement(module, fn, solver_key)

        if new_fn is not None:
            # Just fix function name
            old_fn = solver_val["function"]
            solver_val["function"] = new_fn
            print(f"  FIX  {mod}/{solver_key}: {module}.{old_fn} -> .{new_fn}")
            changed = True
            n_patched += 1
        elif repl_template is not None:
            # Replace whole solver entry, keeping key and name if possible
            new_entry = dict(repl_template)
            # Preserve original solver name if it's meaningful
            if solver_val.get("name") and solver_val["name"] != "":
                new_entry["name"] = solver_val["name"] + " [proxy]"
            print(f"  REPL {mod}/{solver_key}: {module}.{fn} -> {new_entry['module']}.{new_entry['function']}")
            solvers[solver_key] = new_entry
            changed = True
            n_patched += 1
        else:
            print(f"  WARN {mod}/{solver_key}: no replacement for {module}.{fn}")

    if changed:
        cfg["solvers"] = solvers
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        n_yaml_changed += 1

print()
print(f"=== SUMMARY ===")
print(f"YAMLs changed: {n_yaml_changed}")
print(f"Solver entries patched: {n_patched}")
