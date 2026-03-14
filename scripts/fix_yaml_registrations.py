"""Fix YAML solver registrations to use portfolio-compatible run_* wrappers.

Fixes 79 YAML entries across 42 files where direct functions are registered
instead of their (y, physics, cfg) -> (result, info) wrapper equivalents.
"""

import yaml
from pathlib import Path

CONFIGS_DIR = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\benchmarks\configs")

# Map of (module, old_function) -> new_function
# Module stays the same unless noted
FUNCTION_FIXES = {
    # MRI zero-filled
    ("pwm_core.recon.mri_solvers", "zero_filled_reconstruction"): "run_zero_filled",
    # MRI CS wavelet
    ("pwm_core.recon.mri_solvers", "cs_mri_wavelet"): "run_cs_mri",
    # RED-CNN
    ("pwm_core.recon.redcnn", "redcnn_denoise"): "run_redcnn",
    # DeStripe
    ("pwm_core.recon.destripe_net", "destripe_denoise"): "run_destripe",
    # GAP-TV
    ("pwm_core.recon.gap_tv", "gap_tv_cassi"): "run_gap_tv",
    ("pwm_core.recon.gap_tv", "gap_tv_cacti"): "run_gap_tv",
    # EfficientSCI
    ("pwm_core.recon.efficientsci", "efficientsci_recon"): "run_efficientsci",
    # ELP Unfolding
    ("pwm_core.recon.elp_unfolding", "elp_recon"): "run_elp_unfolding",
    # IFCNN
    ("pwm_core.recon.ifcnn", "ifcnn_fuse"): "run_ifcnn",
    # CARE UNet
    ("pwm_core.recon.care_unet", "care_restore_2d"): "run_care",
    ("pwm_core.recon.care_unet", "care_restore_3d"): "run_care",
    # MoDL
    ("pwm_core.recon.modl", "modl_recon"): "run_modl",
    # FLIM
    ("pwm_core.recon.flim_solver", "mle_fit_recon"): "run_flim",
    # PtychoNN
    ("pwm_core.recon.ptychonn", "ptychonn_reconstruct"): "run_ptychonn",
    # SIM
    ("pwm_core.recon.sim_solver", "hifi_sim_2d"): "run_sim_reconstruction",
    ("pwm_core.recon.sim_solver", "fairsim_reconstruction"): "run_sim_reconstruction",
    ("pwm_core.recon.sim_solver", "wiener_sim_2d"): "run_sim_reconstruction",
    # Lightsheet
    ("pwm_core.recon.lightsheet_solver", "vsnr_destripe"): "run_lightsheet",
    # SPC
    ("pwm_core.recon.spc_solvers", "admm_spc"): "run_admm_spc",
    ("pwm_core.recon.spc_solvers", "fista_l1"): "run_fista_l1_spc",
    # LISTA
    ("pwm_core.recon.lista", "lista_reconstruct"): "run_lista",
    # Classical FISTA-L2 (direct function)
    ("pwm_core.recon.classical", "fista_l2"): "run_fista_l2",
}


def fix_yaml(fpath):
    """Fix solver function references in a single YAML file."""
    with open(fpath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "solvers" not in data or not data["solvers"]:
        return 0

    fixes = 0
    for solver_key, solver_cfg in data["solvers"].items():
        if not isinstance(solver_cfg, dict):
            continue
        module = solver_cfg.get("module", "")
        function = solver_cfg.get("function", "")
        key = (module, function)
        if key in FUNCTION_FIXES:
            new_func = FUNCTION_FIXES[key]
            print(f"  {fpath.stem}/{solver_key}: {function} -> {new_func}")
            solver_cfg["function"] = new_func
            fixes += 1

    if fixes > 0:
        with open(fpath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return fixes


def main():
    total = 0
    files_fixed = 0
    for yf in sorted(CONFIGS_DIR.glob("*.yaml")):
        if yf.name.startswith("_"):
            continue
        n = fix_yaml(yf)
        if n > 0:
            files_fixed += 1
            total += n

    print(f"\nDone: {total} solver entries fixed across {files_fixed} YAML files")


if __name__ == "__main__":
    main()
