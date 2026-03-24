#!/bin/bash
# Run all 158 non-flagship modalities, commit & push after each batch of 5
# Usage: bash scripts/run_and_push.sh

set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

MODALITIES=(
  acoustic_emission acoustic_microscopy active_thermography adaptive_optics afm
  angiography asl_mri atom_probe bioluminescence_tomo brachytherapy_img
  brillouin cars cathodoluminescence cest_mri ceus
  clem coded_exposure confocal_3d confocal_endomicroscopy confocal_livecell
  coronagraphy cryo_et ct_fluorescence cup dark_field
  desi dexa dic diffusion_mri digital_breast_tomo
  dna_paint doppler_ultrasound dot ebsd eddy_current
  edx_mapping eels eht_imaging elastography electron_diffraction
  electron_holography electron_tomography endoscopy entangled_photon event_camera
  expansion fib_sem flash_lidar flim fluorescence_microscopy
  fluoroscopy fmri fpm ftir_imaging fundus
  fwi gaussian_splatting ghost_imaging gpr gravitational_wave
  hdr_imaging holography hyperspectral_remote impedance_tomo industrial_ct
  insar integral ism ivus lattice_lightsheet
  libs lidar light_field lightsheet lucky_imaging
  machine_vision magnetic_particle maldi_msi mammography matrix
  mfm minflux mr_elastography mr_fingerprinting mra
  mrs multispectral_sat muon_tomo nerf neutron_diffraction
  neutron_tomo nirs_brain nsom ocean_acoustic_tomo ocean_color
  oct octa odt palm_storm panorama
  particle_calorimetry passive_microwave pet pet_ct pet_mr
  phase_contrast phase_retrieval photoacoustic photometric_stereo polarization
  polsar portal_imaging proton_radiography proton_therapy_img pump_probe
  quantum_illumination radio_astronomy radio_interferometry raman_imaging sar
  saxs seismic_tomo sem shearography shg
  sim sims solar_imaging sonar spect
  spect_ct spectral_ct spinning_disk srs sted
  stem stm streak_camera structured_light swi
  talbot_lau tem terahertz three_photon tirf
  tof_camera two_photon ultrasonic_phased_array us_mri waxs
  weather_radar widefield_lowdose xfel_sfx xray_crystallography xray_ndt
  xray_radiography xrf_imaging xrf_tomo
)

TOTAL=${#MODALITIES[@]}
COUNT=0
BATCH=0

echo "Starting 158-modality run at $(date)"
echo "Total: $TOTAL modalities"

for mod in "${MODALITIES[@]}"; do
  COUNT=$((COUNT + 1))
  BATCH=$((BATCH + 1))
  echo ""
  echo "=== [$COUNT/$TOTAL] $mod ==="

  python scripts/run_158_modalities.py --modality "$mod" 2>&1 || true

  # Commit and push every 5 modalities
  if [ $BATCH -ge 5 ]; then
    echo ""
    echo "--- Committing batch (modalities $((COUNT-4)) to $COUNT) ---"
    git add datasets/templates/index_group*.md datasets/benchmark/algorithm_state.md benchmarks/learn/ scripts/run_158_results.jsonl scripts/run_158_progress.json 2>/dev/null || true
    git commit -m "bench: verify modalities $((COUNT-4))-$COUNT/158 (local server)" 2>/dev/null || true
    git pull --rebase origin master 2>/dev/null || true
    git push origin master 2>/dev/null || true
    BATCH=0
  fi
done

# Final commit for remaining
if [ $BATCH -gt 0 ]; then
  echo ""
  echo "--- Final commit ---"
  git add datasets/templates/index_group*.md datasets/benchmark/algorithm_state.md benchmarks/learn/ scripts/run_158_results.jsonl scripts/run_158_progress.json 2>/dev/null || true
  git commit -m "bench: verify modalities (final batch, local server)" 2>/dev/null || true
  git pull --rebase origin master 2>/dev/null || true
  git push origin master 2>/dev/null || true
fi

echo ""
echo "=== ALL DONE at $(date) ==="
echo "Processed $TOTAL modalities"
