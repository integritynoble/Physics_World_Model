"""Continue batch 1 from where it crashed (after diffusion_mri, mri, fmri)."""
import sys
sys.path.insert(0, "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/scripts")
from build_real_data_batch1 import *

if __name__ == "__main__":
    print("=" * 60)
    print("Batch 1 (continued): remaining modalities")
    print("=" * 60)

    build_mri_variants()  # asl_mri, swi, mra
    build_ct()
    build_ultrasound()
    build_xray_radiography()
    build_oct()
    build_pet()
    build_fluoroscopy()
    build_mammography()
    build_pathology_modalities()  # widefield, confocal_livecell, octa, confocal_endomicroscopy

    print("\n" + "=" * 60)
    print("Batch 1 complete!")
    print("=" * 60)
