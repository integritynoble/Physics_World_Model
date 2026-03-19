"""Fix all remaining wrong category_module assignments in YAML configs."""
import yaml, os

BASE = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/benchmarks/configs"

FIXES = {
    # MRI group → medical_mri_kspace
    "mri.yaml":                {"category_module": "medical_mri_kspace"},
    "fmri.yaml":               {"category_module": "medical_mri_kspace"},
    "diffusion_mri.yaml":      {"category_module": "medical_mri_kspace"},
    "asl_mri.yaml":            {"category_module": "medical_mri_kspace"},
    "cest_mri.yaml":           {"category_module": "medical_mri_kspace"},
    "mr_elastography.yaml":    {"category_module": "medical_mri_kspace"},
    "mr_fingerprinting.yaml":  {"category_module": "medical_mri_kspace"},
    "mra.yaml":                {"category_module": "medical_mri_kspace"},
    "mrs.yaml":                {"category_module": "medical_mri_kspace"},
    "swi.yaml":                {"category_module": "medical_mri_kspace"},
    # Nuclear emission
    "pet.yaml":                {"category_module": "nuclear_emission"},
    "spect.yaml":              {"category_module": "nuclear_emission"},
    # Electron microscopy → electron_ctf
    "cryo_et.yaml":            {"category_module": "electron_ctf"},
    "electron_tomography.yaml": {"category_module": "electron_ctf"},
    # Remote sensing → remote_sensing_sar
    "sar.yaml":                {"category_module": "remote_sensing_sar"},
    "insar.yaml":              {"category_module": "remote_sensing_sar"},
    "polsar.yaml":             {"category_module": "remote_sensing_sar"},
    "radio_astronomy.yaml":    {"category_module": "remote_sensing_sar"},
    "radio_interferometry.yaml": {"category_module": "remote_sensing_sar"},
    "eht_imaging.yaml":        {"category_module": "remote_sensing_sar"},
    # Medical ultrasound / optical → microscopy_psf
    "ceus.yaml":               {"category_module": "microscopy_psf"},
    "doppler_ultrasound.yaml": {"category_module": "microscopy_psf"},
    "elastography.yaml":       {"category_module": "microscopy_psf"},
    "endoscopy.yaml":          {"category_module": "microscopy_psf"},
    "ivus.yaml":               {"category_module": "microscopy_psf"},
    "mammography.yaml":        {"category_module": "medical_ct_radon"},
    "oct.yaml":                {"category_module": "microscopy_psf"},
    "octa.yaml":               {"category_module": "microscopy_psf"},
    "photoacoustic.yaml":      {"category_module": "microscopy_psf"},
    "ultrasound.yaml":         {"category_module": "microscopy_psf"},
    "nirs_brain.yaml":         {"category_module": "microscopy_psf"},
    "dot.yaml":                {"category_module": "microscopy_psf"},
    "confocal_endomicroscopy.yaml": {"category_module": "microscopy_psf"},
    "dexa.yaml":               {"category_module": "medical_ct_radon"},
    # CT-adjacent
    "brachytherapy_img.yaml":  {"category_module": "medical_ct_radon"},
    "muon_tomo.yaml":          {"category_module": "medical_ct_radon"},
    "neutron_tomo.yaml":       {"category_module": "medical_ct_radon"},
    "proton_radiography.yaml": {"category_module": "medical_ct_radon"},
    "proton_therapy_img.yaml": {"category_module": "medical_ct_radon"},
    "xray_crystallography.yaml": {"category_module": "medical_ct_radon"},
    # Scanning probe
    "eddy_current.yaml":       {"category_module": "scanning_probe"},
}

ok = 0
for fname, fields in FIXES.items():
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        print(f"  MISSING: {fname}")
        continue
    with open(fpath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for k, v in fields.items():
        data[k] = v
    with open(fpath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    ok += 1
    print(f"  OK  {fname}: {fields}")

print(f"\nFixed: {ok}/{len(FIXES)}")
