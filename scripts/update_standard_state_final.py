"""Update standard_state.md: mark all 141 newly-built modalities as built-syn."""
import pathlib, re

STATE = pathlib.Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\datasets\benchmark\standard_state.md")
text = STATE.read_text(encoding="utf-8")

# All 141 modalities we just built — change 🔄 building or ❌ pending → 🔧 built-syn
newly_built = [
    "acoustic_emission","acoustic_microscopy","active_thermography","adaptive_optics",
    "afm","angiography","asl_mri","atom_probe","bioluminescence_tomo","brachytherapy_img",
    "brillouin","cars","cathodoluminescence","cbct","cest_mri","ceus","clem","confocal_3d",
    "confocal_endomicroscopy","confocal_livecell","coronagraphy","cryo_et","ct_fluorescence",
    "cup","dark_field","desi","dexa","dic","diffusion_mri","digital_breast_tomo","dna_paint",
    "doppler_ultrasound","dot","ebsd","eddy_current","edx_mapping","eels","elastography",
    "electron_diffraction","electron_holography","electron_tomography","entangled_photon",
    "event_camera","expansion","fib_sem","flash_lidar","flim","fmri","fpm","ftir_imaging",
    "gaussian_splatting","ghost_imaging","gpr","hdr_imaging","impedance_tomo","industrial_ct",
    "insar","integral","ism","ivus","lattice_lightsheet","libs","lidar","light_field",
    "lightsheet","lucky_imaging","magnetic_particle","maldi_msi","mammography","mfm",
    "minflux","mr_elastography","mr_fingerprinting","mra","mrs","multispectral_sat",
    "muon_tomo","neutron_diffraction","neutron_tomo","nirs_brain","nsom",
    "ocean_acoustic_tomo","ocean_color","octa","odt","palm_storm","passive_microwave",
    "pet_ct","pet_mr","phase_contrast","phase_retrieval","photometric_stereo","polarization",
    "polsar","portal_imaging","proton_radiography","proton_therapy_img","ptychography",
    "pump_probe","quantum_illumination","radio_interferometry","raman_imaging","sar","saxs",
    "sem","shearography","shg","sim","sims","sonar","spc","spect","spect_ct","spectral_ct",
    "spinning_disk","srs","sted","stem","stm","streak_camera","structured_light","swi",
    "talbot_lau","tem","terahertz","three_photon","tirf","tof_camera","two_photon",
    "ultrasonic_phased_array","us_mri","waxs","weather_radar","widefield","widefield_lowdose",
    "xfel_sfx","xray_crystallography","xray_ndt","xray_radiography","xrf_imaging","xrf_tomo",
]

changed = 0
for mod in newly_built:
    # Match table row for this modality and change status
    # Pattern: | ... | **mod** | ... | 🔄 building | or | ... | ❌ pending | or | ... | ⚠️ weak-src |
    old_statuses = ["🔄 building", "❌ pending", "⚠️ weak-src"]
    for old_st in old_statuses:
        pattern = f"**{mod}**"
        if pattern in text and old_st in text.split(pattern)[1].split("\n")[0]:
            line_before = text.split(pattern)[0].rsplit("\n", 1)[-1]
            line_after = text.split(pattern)[1].split("\n")[0]
            full_line = line_before + f"**{mod}**" + line_after
            new_line = full_line.replace(f"| {old_st} |", "| 🔧 built-syn |")
            if new_line != full_line:
                text = text.replace(full_line, new_line)
                changed += 1
                break

# Update the header line
text = re.sub(
    r"Last updated:.*",
    "Last updated: 2026-03-13 — 168/168 modalities built | 16 ✅ real-public | 152 🔧 built-synthetic",
    text
)

# Update summary section
old_summary_pattern = r"\| Status \| Count \| Description \|.*?\*\*Coverage:\*\*.*?\n"
new_summary = """| Status | Count | Description |
|--------|-------|-------------|
| ✅ done | 16 | Real public data: cacti, cassi, sd_cassi, spc_kronecker, cryo_em, endoscopy, fundus, eht_imaging, gravitational_wave, hyperspectral_remote, lensless, machine_vision, matrix, nerf, particle_calorimetry, radio_astronomy |
| 🔧 built-syn | 152 | Synthetic data built — needs real public data to upgrade to ✅ |
| 🔄 building | 0 | — |
| ⚠️ weak-src | 0 | — |
| ❌ pending | 0 | — |

**Coverage:** 168/168 (100%) have built standard datasets · 16/168 (9.5%) verified with real public data
"""

# Find and replace summary block
lines = text.split("\n")
summary_start = None
summary_end = None
for i, line in enumerate(lines):
    if "| Status | Count | Description |" in line:
        summary_start = i
    if summary_start and "**Coverage:**" in line:
        summary_end = i + 1
        break

if summary_start is not None and summary_end is not None:
    text = "\n".join(lines[:summary_start]) + "\n" + new_summary + "\n".join(lines[summary_end+1:])

STATE.write_text(text, encoding="utf-8")
print(f"Updated {changed} modality statuses to built-syn")
print("Header and summary updated.")
print("Done.")
