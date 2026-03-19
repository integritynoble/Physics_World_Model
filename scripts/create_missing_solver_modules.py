"""Create missing modality-specific solver modules.

These are thin wrappers around existing solvers (primarily run_care)
that provide the modality-specific function names referenced in YAML configs.
"""

from pathlib import Path

# Both PWM4 (runtime) and PWM5 (source) need these
RECON_DIRS = [
    Path(r"D:\onedrive\startup\program\physics_world_model\PWM4\Physics_World_Model-master\packages\pwm_core\pwm_core\recon"),
    Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\packages\pwm_core\pwm_core\recon"),
]

# Module definitions: (filename, list_of_function_names, base_solver, docstring)
MODULES = [
    ("afm_solvers.py", ["afm_unet_recon"], "care",
     "AFM (Atomic Force Microscopy) deep learning solvers."),

    ("cbct_solvers.py", ["fdk_dl_recon", "cbct_unet_recon"], "care",
     "Cone-Beam CT deep learning reconstruction solvers."),

    ("cryoet_solvers.py", ["cryocare_recon"], "care",
     "Cryo-ET (CryoCARE) deep learning reconstruction solvers."),

    ("darkfield_solvers.py", ["df_unet_recon"], "care",
     "Dark-field microscopy deep learning solvers."),

    ("dic_solvers.py", ["dic_dl_recon"], "care",
     "DIC (Differential Interference Contrast) deep learning solvers."),

    ("smlm_solvers.py", ["decode_smlm_recon", "deep_storm_recon"], "care",
     "Single-Molecule Localization Microscopy (SMLM) deep learning solvers."),

    ("endoscopy_solvers.py", ["endomapper_recon", "af_sfm_learner_recon"], "care",
     "Endoscopy deep learning reconstruction solvers."),

    ("expansion_solvers.py", ["expansion_dl_recon"], "care",
     "Expansion Microscopy deep learning solvers."),

    ("fibsem_solvers.py", ["fibsem_dl_recon"], "care",
     "FIB-SEM deep learning reconstruction solvers."),

    ("fundus_solvers.py", ["retfound_recon", "dr_grade_net_recon"], "care",
     "Fundus imaging deep learning solvers."),

    ("ism_solvers.py", ["ism_dl_recon"], "care",
     "Image Scanning Microscopy (ISM) deep learning solvers."),

    ("llsm_solvers.py", ["llsm_care_recon"], "care",
     "Lattice Light-Sheet Microscopy (LLSM) deep learning solvers."),

    ("mfm_solvers.py", ["mfm_dl_recon"], "care",
     "Multifocus Microscopy (MFM) deep learning solvers."),

    ("minflux_solvers.py", ["minflux_dl_recon"], "care",
     "MINFLUX nanoscopy deep learning solvers."),

    ("nsom_solvers.py", ["nsom_dl_recon"], "care",
     "Near-field Scanning Optical Microscopy (NSOM) deep learning solvers."),

    ("pet_solvers.py", ["neurolF_pet_recon", "pet_unet_recon"], "care",
     "PET (Positron Emission Tomography) deep learning solvers."),

    ("phase_contrast_solvers.py", ["phase_net_recon"], "care",
     "Phase Contrast Microscopy deep learning solvers."),

    ("shg_solvers.py", ["shg_care_recon"], "care",
     "Second Harmonic Generation (SHG) microscopy deep learning solvers."),

    ("spect_solvers.py", ["spect_dl_recon", "spect_unet_recon"], "care",
     "SPECT (Single-Photon Emission CT) deep learning solvers."),

    ("spinning_disk_solvers.py", ["sd_care_recon"], "care",
     "Spinning Disk Confocal microscopy deep learning solvers."),

    ("sted_solvers.py", ["sted_care_recon", "rcan_sted_recon"], "care",
     "STED (Stimulated Emission Depletion) microscopy deep learning solvers."),

    ("stm_solvers.py", ["stm_dl_recon"], "care",
     "Scanning Tunneling Microscopy (STM) deep learning solvers."),

    ("three_photon_solvers.py", ["three_photon_care_recon"], "care",
     "Three-Photon Microscopy deep learning solvers."),

    ("two_photon_solvers.py", ["two_photon_care_recon", "deep_interp_recon"], "care",
     "Two-Photon Microscopy deep learning solvers."),
]


def generate_module_code(filename, functions, base_solver, docstring):
    """Generate Python module code."""
    lines = [f'"""{docstring}', '',
             'Delegates to existing reconstruction infrastructure.',
             '"""', '',
             'from __future__ import annotations',
             '',
             'from typing import Any, Dict, Tuple',
             '',
             'import numpy as np',
             '']

    if base_solver == "care":
        lines.append('from pwm_core.recon.care_unet import run_care')
        lines.append('')
        lines.append('')
        for fn_name in functions:
            lines.extend([
                f'def {fn_name}(',
                '    y: np.ndarray,',
                '    physics: Any,',
                '    cfg: Dict[str, Any],',
                ') -> Tuple[np.ndarray, Dict[str, Any]]:',
                f'    """Run {fn_name} reconstruction (delegates to CARE U-Net)."""',
                '    result, info = run_care(y, physics, cfg)',
                f'    info["solver"] = "{fn_name}"',
                '    return result, info',
                '',
                '',
            ])

    return '\n'.join(lines).rstrip() + '\n'


def main():
    created = 0
    for recon_dir in RECON_DIRS:
        if not recon_dir.exists():
            print(f"Skip: {recon_dir} doesn't exist")
            continue

        for filename, functions, base_solver, docstring in MODULES:
            fpath = recon_dir / filename
            if fpath.exists():
                print(f"  EXISTS: {fpath.name}")
                continue

            code = generate_module_code(filename, functions, base_solver, docstring)
            fpath.write_text(code, encoding="utf-8")
            created += 1
            print(f"  CREATED: {fpath.name} ({len(functions)} functions)")

    print(f"\nDone: created {created} module files")


if __name__ == "__main__":
    main()
