# Modality Templates Index

Comprehensive 7-step templates for all 168 imaging modalities in the PWM5 Benchmark.
Each template covers: (1) Verify Standard Dataset, (2) List All Algorithms, (3) Update Solvers, (4) Verify Each Algorithm, (5) Upload Checkpoints to GCS, (6) Upload Standard Dataset to GCS, (7) Push to GitHub.

---

## Template Files in This Folder (55 modalities)

These templates cover all modalities that were not already in `_templates_part1.md` through `_templates_part8.md`.

### [medical_core.md](medical_core.md) — Medical Imaging Core (7 modalities)

| # | Modality ID | Name |
|---|-------------|------|
| 1 | `mri` | Magnetic Resonance Imaging |
| 2 | `ct` | X-ray Computed Tomography |
| 3 | `ct_fluorescence` | X-ray Fluorescence CT |
| 4 | `pet_ct` | PET-CT Combined |
| 5 | `pet_mr` | PET-MR Combined |
| 6 | `spect_ct` | SPECT-CT Combined |
| 7 | `us_mri` | Ultrasound-MRI Fusion |

### [remote_sensing.md](remote_sensing.md) — Remote Sensing & Radar (10 modalities)

| # | Modality ID | Name |
|---|-------------|------|
| 1 | `sar` | Synthetic Aperture Radar |
| 2 | `polsar` | Polarimetric SAR |
| 3 | `insar` | Interferometric SAR |
| 4 | `gpr` | Ground-Penetrating Radar |
| 5 | `hyperspectral_remote` | Hyperspectral Remote Sensing |
| 6 | `multispectral_sat` | Multispectral Satellite Imaging |
| 7 | `ocean_color` | Ocean Color Remote Sensing |
| 8 | `passive_microwave` | Passive Microwave Sensing |
| 9 | `weather_radar` | Weather Radar |
| 10 | `sonar` | Sonar Imaging |

### [scanning_probe_spectroscopy.md](scanning_probe_spectroscopy.md) — Scanning Probe & Spectroscopic Imaging (13 modalities)

| # | Modality ID | Name |
|---|-------------|------|
| 1 | `afm` | Atomic Force Microscopy |
| 2 | `stm` | Scanning Tunneling Microscopy |
| 3 | `mfm` | Magnetic Force Microscopy |
| 4 | `nsom` | Near-field Scanning Optical Microscopy |
| 5 | `raman_imaging` | Raman Imaging / Microscopy |
| 6 | `srs` | Stimulated Raman Scattering |
| 7 | `cars` | Coherent Anti-Stokes Raman Scattering |
| 8 | `brillouin` | Brillouin Microscopy |
| 9 | `ftir_imaging` | FTIR Imaging |
| 10 | `libs` | Laser-Induced Breakdown Spectroscopy |
| 11 | `maldi_msi` | MALDI Mass Spectrometry Imaging |
| 12 | `desi` | DESI Mass Spectrometry Imaging |
| 13 | `sims` | Secondary Ion Mass Spectrometry |

### [electron_nuclear_xray.md](electron_nuclear_xray.md) — Electron Microscopy, Nuclear & X-ray Techniques (13 modalities)

| # | Modality ID | Name |
|---|-------------|------|
| 1 | `cryo_em` | Single-Particle Cryo-Electron Microscopy |
| 2 | `cathodoluminescence` | Cathodoluminescence |
| 3 | `clem` | Correlative Light-Electron Microscopy |
| 4 | `atom_probe` | Atom Probe Tomography |
| 5 | `muon_tomo` | Muon Tomography |
| 6 | `neutron_tomo` | Neutron Tomography |
| 7 | `neutron_diffraction` | Neutron Diffraction |
| 8 | `proton_radiography` | Proton Radiography |
| 9 | `saxs` | Small-Angle X-ray Scattering |
| 10 | `waxs` | Wide-Angle X-ray Scattering |
| 11 | `xfel_sfx` | XFEL Serial Femtosecond Crystallography |
| 12 | `xray_crystallography` | X-ray Crystallography |
| 13 | `xrf_tomo` | XRF Tomography |

### [quantum_3d_compressive.md](quantum_3d_compressive.md) — Quantum, 3D Reconstruction & Compressive/Ultrafast (12 modalities)

| # | Modality ID | Name |
|---|-------------|------|
| 1 | `ghost_imaging` | Ghost Imaging |
| 2 | `quantum_illumination` | Quantum Illumination |
| 3 | `entangled_photon` | Entangled Photon Imaging |
| 4 | `nerf` | Neural Radiance Fields |
| 5 | `gaussian_splatting` | 3D Gaussian Splatting |
| 6 | `cup` | Compressed Ultrafast Photography |
| 7 | `sd_cassi` | Single-Disperser CASSI |
| 8 | `spc_block` | Block-Diagonal Single-Pixel Camera |
| 9 | `spc_kronecker` | Kronecker Product SPC |
| 10 | `streak_camera` | Streak Camera |
| 11 | `pump_probe` | Pump-Probe Spectroscopy / Imaging |
| 12 | `radio_interferometry` | Radio Interferometry |

---

## Previously Written Templates (113 modalities in `_templates_part1.md` — `_templates_part8.md`)

### `_templates_part1.md` — Astronomy & Space Imaging (4 modalities)
- `coronagraphy`, `eht_imaging`, `lucky_imaging`, `solar_imaging`

### `_templates_part2.md` — Broader Experimental Science (11 modalities)
- `acoustic_emission`, `adaptive_optics`, `bioluminescence_tomo`, `fwi`, `gravitational_wave`, `impedance_tomo`, `magnetic_particle`, `ocean_acoustic_tomo`, `particle_calorimetry`, `radio_astronomy`, `seismic_tomo`

### `_templates_part3.md` — Coherent Imaging + Compressive Imaging (9 modalities)
- `holography`, `odt`, `phase_retrieval`, `ptychography`, `talbot_lau`, `cacti`, `cassi`, `matrix`, `spc`

### `_templates_part4.md` — Computational Optics + Photography + Depth Imaging (12 modalities)
- `integral`, `light_field`, `coded_exposure`, `event_camera`, `hdr_imaging`, `lensless`, `panorama`, `flash_lidar`, `lidar`, `photometric_stereo`, `structured_light`, `tof_camera`

### `_templates_part5.md` — Electron Microscopy + Industrial Inspection (21 modalities)
- `cryo_et`, `ebsd`, `edx_mapping`, `eels`, `electron_diffraction`, `electron_holography`, `electron_tomography`, `fib_sem`, `sem`, `stem`, `tem`, `acoustic_microscopy`, `active_thermography`, `eddy_current`, `industrial_ct`, `machine_vision`, `shearography`, `terahertz`, `ultrasonic_phased_array`, `xray_ndt`, `xrf_imaging`

### `_templates_part6.md` — Medical Imaging Part 1 (20 modalities)
- `angiography`, `asl_mri`, `brachytherapy_img`, `cbct`, `cest_mri`, `ceus`, `confocal_endomicroscopy`, `dexa`, `diffusion_mri`, `digital_breast_tomo`, `doppler_ultrasound`, `dot`, `elastography`, `endoscopy`, `fluoroscopy`, `fmri`, `fundus`, `ivus`, `mammography`, `mr_elastography`

### `_templates_part7.md` — Medical Imaging Part 2 (15 modalities)
- `mr_fingerprinting`, `mra`, `mrs`, `nirs_brain`, `oct`, `octa`, `pet`, `photoacoustic`, `portal_imaging`, `proton_therapy_img`, `spect`, `spectral_ct`, `swi`, `ultrasound`, `xray_radiography`

### `_templates_part8.md` — Microscopy (24 modalities)
- `confocal_3d`, `confocal_livecell`, `dark_field`, `dic`, `dna_paint`, `expansion`, `flim`, `fpm`, `ism`, `lattice_lightsheet`, `lightsheet`, `minflux`, `palm_storm`, `phase_contrast`, `polarization`, `shg`, `sim`, `spinning_disk`, `sted`, `three_photon`, `tirf`, `two_photon`, `widefield`, `widefield_lowdose`

---

## Summary

| Location | Files | Modalities | Steps |
|----------|-------|------------|-------|
| `templates/` (this folder) | 5 | 55 | 385 |
| `_templates_part1–8.md` (parent) | 8 | 113 | 791 |
| **Total** | **13** | **168** | **1176** |

All 168 modalities now have complete 7-step templates.
