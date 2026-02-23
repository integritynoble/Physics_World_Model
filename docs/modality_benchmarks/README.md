# PWM Per-Modality Benchmark Files

> **168 modalities** x **4 benchmarks** (B1-B4) x **5 maturity levels** (M0-M4)
>
> Each file defines the complete benchmark roadmap for one imaging modality,
> from template (M0) to adversarial (M4).

---

## Summary

| # | Category | Count | Validated | Registered | Planned |
|---|----------|------:|----------:|-----------:|--------:|
| 1 | Microscopy | 24 | 0 | 24 | 0 |
| 2 | Compressive Imaging | 4 | 3 | 1 | 0 |
| 3 | Medical Imaging | 37 | 2 | 31 | 4 |
| 4 | Coherent Imaging | 5 | 1 | 4 | 0 |
| 5 | Computational Photography | 5 | 1 | 4 | 0 |
| 6 | Computational Optics | 2 | 0 | 2 | 0 |
| 7 | Neural Rendering | 2 | 0 | 2 | 0 |
| 8 | Electron Microscopy | 11 | 0 | 11 | 0 |
| 9 | Depth Imaging | 5 | 0 | 5 | 0 |
| 10 | Remote Sensing | 11 | 0 | 10 | 1 |
| 11 | Industrial Inspection | 10 | 0 | 10 | 0 |
| 12 | Scientific Instrumentation | 12 | 0 | 12 | 0 |
| 13 | Broader Experimental Science | 11 | 0 | 9 | 2 |
| 14 | Spectroscopy & Spectral Imaging | 8 | 0 | 8 | 0 |
| 15 | Ultrafast Imaging | 4 | 0 | 4 | 0 |
| 16 | Quantum Imaging | 3 | 0 | 1 | 2 |
| 17 | Multi-Modal Fusion | 6 | 0 | 6 | 0 |
| 18 | Scanning Probe Microscopy | 4 | 0 | 4 | 0 |
| 19 | Astronomy & Space Imaging | 4 | 0 | 4 | 0 |
| | **Total** | **168** | **7** | **152** | **9** |

---

## 1. Microscopy (24 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 1 | Widefield Fluorescence Microscopy | C --> D | M3 | Registered | [widefield.md](widefield.md) |
| 2 | Low-Dose Widefield Microscopy | C --> D | M1 | Registered | [widefield_lowdose.md](widefield_lowdose.md) |
| 3 | Confocal Live-Cell Microscopy | C --> D | M1 | Registered | [confocal_livecell.md](confocal_livecell.md) |
| 4 | Confocal 3D Z-Stack | C --> D | M1 | Registered | [confocal_3d.md](confocal_3d.md) |
| 5 | Structured Illumination Microscopy (SIM) | M --> C --> D | M2 | Registered | [sim.md](sim.md) |
| 6 | Light-Sheet Fluorescence Microscopy (LSFM) | C --> D | M1 | Registered | [lightsheet.md](lightsheet.md) |
| 7 | Fluorescence Lifetime Imaging (FLIM) | M --> R --> D | M0 | Registered | [flim.md](flim.md) |
| 8 | Fourier Ptychographic Microscopy (FPM) | M --> P --> D | M1 | Registered | [fpm.md](fpm.md) |
| 9 | Two-Photon / Multiphoton Microscopy | C --> D | M0 | Registered | [two_photon.md](two_photon.md) |
| 10 | STED Microscopy | C --> D | M0 | Registered | [sted.md](sted.md) |
| 11 | PALM/STORM Single-Molecule Localization | M --> D | M0 | Registered | [palm_storm.md](palm_storm.md) |
| 12 | TIRF Microscopy | C --> D | M0 | Registered | [tirf.md](tirf.md) |
| 13 | Polarization Microscopy | M --> C --> D | M0 | Registered | [polarization.md](polarization.md) |
| 14 | Expansion Microscopy (ExM) | C --> D | M0 | Registered | [expansion.md](expansion.md) |
| 15 | MINFLUX Nanoscopy | C --> D | M0 | Registered | [minflux.md](minflux.md) |
| 16 | Image Scanning Microscopy (ISM) | C --> D | M0 | Registered | [ism.md](ism.md) |
| 17 | Phase Contrast Microscopy | C --> D | M0 | Registered | [phase_contrast.md](phase_contrast.md) |
| 18 | Differential Interference Contrast (DIC) | M --> C --> D | M0 | Registered | [dic.md](dic.md) |
| 19 | Dark-Field Microscopy | C --> D | M0 | Registered | [dark_field.md](dark_field.md) |
| 20 | Lattice Light-Sheet Microscopy | C --> D | M0 | Registered | [lattice_lightsheet.md](lattice_lightsheet.md) |
| 21 | Second Harmonic Generation (SHG) Microscopy | M --> R --> D | M0 | Registered | [shg.md](shg.md) |
| 22 | Spinning Disk Confocal Microscopy | C --> D | M0 | Registered | [spinning_disk.md](spinning_disk.md) |
| 23 | Three-Photon Microscopy | C --> D | M0 | Registered | [three_photon.md](three_photon.md) |
| 24 | DNA-PAINT Super-Resolution | M --> D | M0 | Registered | [dna_paint.md](dna_paint.md) |

---

## 2. Compressive Imaging (4 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 25 | Coded Aperture Snapshot Spectral Imaging (CASSI) | M --> W --> Sigma --> D | M3 | Validated | [cassi.md](cassi.md) |
| 26 | Single-Pixel Camera (SPC) | M --> Sigma --> D | M3 | Validated | [spc.md](spc.md) |
| 27 | Coded Aperture Compressive Temporal Imaging (CACTI) | M --> Sigma --> D | M3 | Validated | [cacti.md](cacti.md) |
| 28 | Generic Matrix Sensing | M --> D | M1 | Registered | [matrix.md](matrix.md) |

---

## 3. Medical Imaging (37 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 29 | X-ray Computed Tomography (CT) | Pi --> D | M3 | Validated | [ct.md](ct.md) |
| 30 | Magnetic Resonance Imaging (MRI) | M --> F --> S --> D | M3 | Validated | [mri.md](mri.md) |
| 31 | X-ray Radiography | Pi --> D | M0 | Registered | [xray_radiography.md](xray_radiography.md) |
| 32 | Ultrasound B-mode Imaging | P --> D | M1 | Planned | [ultrasound.md](ultrasound.md) |
| 33 | Positron Emission Tomography (PET) | Pi --> D | M1 | Registered | [pet.md](pet.md) |
| 34 | Single Photon Emission CT (SPECT) | Pi --> D | M0 | Registered | [spect.md](spect.md) |
| 35 | Fluoroscopy | Pi --> D | M0 | Registered | [fluoroscopy.md](fluoroscopy.md) |
| 36 | Mammography | Pi --> D | M0 | Registered | [mammography.md](mammography.md) |
| 37 | Dual-Energy X-ray Absorptiometry (DEXA) | Pi --> D | M0 | Registered | [dexa.md](dexa.md) |
| 38 | Cone-Beam Computed Tomography (CBCT) | Pi --> D | M0 | Registered | [cbct.md](cbct.md) |
| 39 | X-ray Angiography | Pi --> D | M0 | Registered | [angiography.md](angiography.md) |
| 40 | Diffuse Optical Tomography (DOT) | M --> R,P,R --> D | M0 | Registered | [dot.md](dot.md) |
| 41 | Photoacoustic Imaging | M --> P --> D | M0 | Planned | [photoacoustic.md](photoacoustic.md) |
| 42 | Optical Coherence Tomography (OCT) | P+P --> Sigma --> D | M1 | Planned | [oct.md](oct.md) |
| 43 | Functional MRI (BOLD fMRI) | M --> F --> S --> D | M0 | Registered | [fmri.md](fmri.md) |
| 44 | MR Spectroscopy (MRS) | M --> F --> S --> D | M0 | Registered | [mrs.md](mrs.md) |
| 45 | Diffusion MRI (DTI) | M --> F --> S --> D | M0 | Registered | [diffusion_mri.md](diffusion_mri.md) |
| 46 | Doppler Ultrasound | P --> D | M0 | Registered | [doppler_ultrasound.md](doppler_ultrasound.md) |
| 47 | Shear-Wave Elastography | P --> D | M0 | Registered | [elastography.md](elastography.md) |
| 48 | Fiber Bundle Endoscopy | M --> C --> D | M0 | Registered | [endoscopy.md](endoscopy.md) |
| 49 | Fundus Camera | C --> D | M0 | Registered | [fundus.md](fundus.md) |
| 50 | OCT Angiography (OCTA) | P+P --> Sigma --> D | M0 | Registered | [octa.md](octa.md) |
| 51 | Proton Therapy Imaging | Pi --> D | M0 | Registered | [proton_therapy_img.md](proton_therapy_img.md) |
| 52 | Brachytherapy Imaging | Pi --> D | M0 | Registered | [brachytherapy_img.md](brachytherapy_img.md) |
| 53 | Portal Imaging (EPID) | Pi --> D | M0 | Registered | [portal_imaging.md](portal_imaging.md) |
| 54 | Photon-Counting Spectral CT | Pi --> W --> D | M0 | Planned | [spectral_ct.md](spectral_ct.md) |
| 55 | MR Elastography (MRE) | M --> F --> S --> D | M0 | Registered | [mr_elastography.md](mr_elastography.md) |
| 56 | CEST MRI | M --> F --> S --> D | M0 | Registered | [cest_mri.md](cest_mri.md) |
| 57 | Arterial Spin Labeling (ASL) MRI | M --> F --> S --> D | M0 | Registered | [asl_mri.md](asl_mri.md) |
| 58 | MR Angiography (MRA) | M --> F --> S --> D | M0 | Registered | [mra.md](mra.md) |
| 59 | Susceptibility-Weighted Imaging (SWI) | M --> F --> S --> D | M0 | Registered | [swi.md](swi.md) |
| 60 | MR Fingerprinting (MRF) | M --> F --> S --> D | M0 | Registered | [mr_fingerprinting.md](mr_fingerprinting.md) |
| 61 | Intravascular Ultrasound (IVUS) | P --> D | M0 | Registered | [ivus.md](ivus.md) |
| 62 | Contrast-Enhanced Ultrasound (CEUS) | P --> R --> D | M0 | Registered | [ceus.md](ceus.md) |
| 63 | Digital Breast Tomosynthesis (DBT) | Pi --> D | M0 | Registered | [digital_breast_tomo.md](digital_breast_tomo.md) |
| 64 | Confocal Laser Endomicroscopy (CLE) | M --> C --> D | M0 | Registered | [confocal_endomicroscopy.md](confocal_endomicroscopy.md) |
| 65 | Functional Near-Infrared Spectroscopy (fNIRS) | M --> R,P --> D | M0 | Registered | [nirs_brain.md](nirs_brain.md) |

---

## 4. Coherent Imaging (5 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 66 | Ptychographic Imaging | M --> P --> D | M3 | Validated | [ptychography.md](ptychography.md) |
| 67 | Digital Holographic Microscopy | P --> D | M1 | Registered | [holography.md](holography.md) |
| 68 | Coherent Diffractive Imaging / Phase Retrieval | P --> D | M0 | Registered | [phase_retrieval.md](phase_retrieval.md) |
| 69 | Optical Diffraction Tomography (ODT) | P --> D | M0 | Registered | [odt.md](odt.md) |
| 70 | Talbot-Lau X-ray Grating Interferometry | M --> P --> D | M0 | Registered | [talbot_lau.md](talbot_lau.md) |

---

## 5. Computational Photography (5 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 71 | Lensless (Diffuser Camera) Imaging | C --> D | M3 | Validated | [lensless.md](lensless.md) |
| 72 | Panorama Multi-Focus Fusion | C --> D | M0 | Registered | [panorama.md](panorama.md) |
| 73 | Coded Exposure / Flutter Shutter | M --> C --> D | M0 | Registered | [coded_exposure.md](coded_exposure.md) |
| 74 | Event Camera / Dynamic Vision Sensor (DVS) | M --> D | M0 | Registered | [event_camera.md](event_camera.md) |
| 75 | High Dynamic Range (HDR) Imaging | M --> Sigma --> D | M0 | Registered | [hdr_imaging.md](hdr_imaging.md) |

---

## 6. Computational Optics (2 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 76 | Light Field Imaging | C --> S --> D | M0 | Registered | [light_field.md](light_field.md) |
| 77 | Integral Photography | C --> S --> D | M0 | Registered | [integral.md](integral.md) |

---

## 7. Neural Rendering (2 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 78 | Neural Radiance Fields (NeRF) | M --> P --> D | M0 | Registered | [nerf.md](nerf.md) |
| 79 | 3D Gaussian Splatting (3DGS) | M --> P --> D | M0 | Registered | [gaussian_splatting.md](gaussian_splatting.md) |

---

## 8. Electron Microscopy (11 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 80 | Scanning Electron Microscopy (SEM) | C --> D | M0 | Registered | [sem.md](sem.md) |
| 81 | Transmission Electron Microscopy (TEM) | C --> D | M0 | Registered | [tem.md](tem.md) |
| 82 | Electron Tomography | Pi --> D | M0 | Registered | [electron_tomography.md](electron_tomography.md) |
| 83 | Scanning Transmission Electron Microscopy (STEM) | S --> D | M0 | Registered | [stem.md](stem.md) |
| 84 | 4D-STEM Electron Diffraction | M --> P --> D | M0 | Registered | [electron_diffraction.md](electron_diffraction.md) |
| 85 | Electron Backscatter Diffraction (EBSD) | R --> D | M0 | Registered | [ebsd.md](ebsd.md) |
| 86 | Electron Energy Loss Spectroscopy (EELS) | S --> D | M0 | Registered | [eels.md](eels.md) |
| 87 | Electron Holography | P --> D | M0 | Registered | [electron_holography.md](electron_holography.md) |
| 88 | Cryo-Electron Tomography (Cryo-ET) | Pi --> D | M0 | Registered | [cryo_et.md](cryo_et.md) |
| 89 | Focused Ion Beam SEM (FIB-SEM) | S --> C --> D | M0 | Registered | [fib_sem.md](fib_sem.md) |
| 90 | STEM-EDX Elemental Mapping | M --> R --> D | M0 | Registered | [edx_mapping.md](edx_mapping.md) |

---

## 9. Depth Imaging (5 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 91 | Time-of-Flight Depth Camera | P --> D | M0 | Registered | [tof_camera.md](tof_camera.md) |
| 92 | LiDAR Scanner | P --> S --> D | M0 | Registered | [lidar.md](lidar.md) |
| 93 | Structured-Light Depth Camera | M --> C --> D | M0 | Registered | [structured_light.md](structured_light.md) |
| 94 | Photometric Stereo | M --> C --> D | M0 | Registered | [photometric_stereo.md](photometric_stereo.md) |
| 95 | Flash LiDAR | P --> D | M0 | Registered | [flash_lidar.md](flash_lidar.md) |

---

## 10. Remote Sensing (11 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 96 | Synthetic Aperture Radar (SAR) | F --> D | M0 | Planned | [sar.md](sar.md) |
| 97 | Sonar Imaging | P --> D | M0 | Registered | [sonar.md](sonar.md) |
| 98 | Hyperspectral Remote Sensing | M --> W --> Sigma --> D | M0 | Registered | [hyperspectral_remote.md](hyperspectral_remote.md) |
| 99 | Multispectral Satellite Imaging | M --> Sigma --> D | M0 | Registered | [multispectral_sat.md](multispectral_sat.md) |
| 100 | Ground-Penetrating Radar (GPR) | P --> D | M0 | Registered | [gpr.md](gpr.md) |
| 101 | Weather / Doppler Radar | P --> R --> D | M0 | Registered | [weather_radar.md](weather_radar.md) |
| 102 | Radio Interferometry (VLBI) | F --> S --> D | M0 | Registered | [radio_interferometry.md](radio_interferometry.md) |
| 103 | Passive Microwave Radiometry | Sigma --> D | M0 | Registered | [passive_microwave.md](passive_microwave.md) |
| 104 | Interferometric SAR (InSAR) | F --> S --> D | M0 | Registered | [insar.md](insar.md) |
| 105 | Polarimetric SAR (PolSAR) | F --> M --> D | M0 | Registered | [polsar.md](polsar.md) |
| 106 | Ocean Color Remote Sensing | M --> Sigma --> D | M0 | Registered | [ocean_color.md](ocean_color.md) |

---

## 11. Industrial Inspection (10 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 107 | Industrial X-ray CT | Pi --> D | M0 | Registered | [industrial_ct.md](industrial_ct.md) |
| 108 | X-ray NDT (Radiography) | Pi --> D | M0 | Registered | [xray_ndt.md](xray_ndt.md) |
| 109 | Ultrasonic Phased Array (TFM/FMC) | P --> D | M0 | Registered | [ultrasonic_phased_array.md](ultrasonic_phased_array.md) |
| 110 | Eddy Current Imaging | F --> D | M0 | Registered | [eddy_current.md](eddy_current.md) |
| 111 | Active Thermography (IR) | P --> D | M0 | Registered | [active_thermography.md](active_thermography.md) |
| 112 | Terahertz Imaging (THz) | P --> D | M0 | Registered | [terahertz.md](terahertz.md) |
| 113 | Machine Vision / AOI | C --> D | M0 | Registered | [machine_vision.md](machine_vision.md) |
| 114 | X-ray Fluorescence (XRF) Imaging | M --> R --> D | M0 | Registered | [xrf_imaging.md](xrf_imaging.md) |
| 115 | Shearography | M --> P --> D | M0 | Registered | [shearography.md](shearography.md) |
| 116 | Scanning Acoustic Microscopy (SAM) | P --> D | M0 | Registered | [acoustic_microscopy.md](acoustic_microscopy.md) |

---

## 12. Scientific Instrumentation (12 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 117 | X-ray Crystallography | F --> S --> D | M0 | Registered | [xray_crystallography.md](xray_crystallography.md) |
| 118 | Small-Angle X-ray Scattering (SAXS) | R --> D | M0 | Registered | [saxs.md](saxs.md) |
| 119 | MALDI Mass Spectrometry Imaging | S --> D | M0 | Registered | [maldi_msi.md](maldi_msi.md) |
| 120 | Atom Probe Tomography (APT) | S --> D | M0 | Registered | [atom_probe.md](atom_probe.md) |
| 121 | Cryo-EM Single Particle Analysis | C --> D | M0 | Registered | [cryo_em.md](cryo_em.md) |
| 122 | Neutron Radiography / Tomography | Pi --> D | M0 | Registered | [neutron_tomo.md](neutron_tomo.md) |
| 123 | Proton Radiography | Pi --> D | M0 | Registered | [proton_radiography.md](proton_radiography.md) |
| 124 | Muon Tomography | Pi --> D | M0 | Registered | [muon_tomo.md](muon_tomo.md) |
| 125 | Wide-Angle X-ray Scattering (WAXS) | R --> D | M0 | Registered | [waxs.md](waxs.md) |
| 126 | X-ray Fluorescence Tomography | Pi --> R --> D | M0 | Registered | [xrf_tomo.md](xrf_tomo.md) |
| 127 | Neutron Diffraction | R --> S --> D | M0 | Registered | [neutron_diffraction.md](neutron_diffraction.md) |
| 128 | Cathodoluminescence (CL) Imaging | M --> R --> D | M0 | Registered | [cathodoluminescence.md](cathodoluminescence.md) |

---

## 13. Broader Experimental Science (11 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 129 | Adaptive Optics (AO) Imaging | M --> C --> D | M0 | Registered | [adaptive_optics.md](adaptive_optics.md) |
| 130 | Seismic Tomography | P --> D | M0 | Registered | [seismic_tomo.md](seismic_tomo.md) |
| 131 | Gravitational Wave Detection | P --> Sigma --> D | M0 | Registered | [gravitational_wave.md](gravitational_wave.md) |
| 132 | Particle Calorimetry | R --> Sigma --> D | M0 | Registered | [particle_calorimetry.md](particle_calorimetry.md) |
| 133 | Radio Aperture Synthesis | F --> S --> D | M0 | Registered | [radio_astronomy.md](radio_astronomy.md) |
| 134 | Acoustic Emission Testing (AE) | P --> S --> D | M0 | Registered | [acoustic_emission.md](acoustic_emission.md) |
| 135 | Magnetic Particle Imaging (MPI) | M --> F --> D | M0 | Registered | [magnetic_particle.md](magnetic_particle.md) |
| 136 | Electrical Impedance Tomography (EIT) | M --> D | M0 | Registered | [impedance_tomo.md](impedance_tomo.md) |
| 137 | Full-Waveform Inversion (FWI) | P --> D | M0 | Planned | [fwi.md](fwi.md) |
| 138 | Ocean Acoustic Tomography | P --> D | M0 | Planned | [ocean_acoustic_tomo.md](ocean_acoustic_tomo.md) |
| 139 | Bioluminescence Tomography (BLT) | Src --> R,P --> D | M0 | Registered | [bioluminescence_tomo.md](bioluminescence_tomo.md) |

---

## 14. Spectroscopy & Spectral Imaging (8 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 140 | Raman Imaging / Microscopy | M --> R --> D | M0 | Registered | [raman_imaging.md](raman_imaging.md) |
| 141 | Coherent Anti-Stokes Raman (CARS) Microscopy | M --> R --> D | M0 | Registered | [cars.md](cars.md) |
| 142 | Stimulated Raman Scattering (SRS) Microscopy | M --> R --> D | M0 | Registered | [srs.md](srs.md) |
| 143 | FTIR Spectroscopic Imaging | M --> Sigma --> D | M0 | Registered | [ftir_imaging.md](ftir_imaging.md) |
| 144 | Laser-Induced Breakdown Spectroscopy (LIBS) Imaging | M --> R --> D | M0 | Registered | [libs.md](libs.md) |
| 145 | Brillouin Microscopy | M --> R --> D | M0 | Registered | [brillouin.md](brillouin.md) |
| 146 | Secondary Ion Mass Spectrometry (SIMS) Imaging | S --> D | M0 | Registered | [sims.md](sims.md) |
| 147 | DESI Mass Spectrometry Imaging | S --> D | M0 | Registered | [desi.md](desi.md) |

---

## 15. Ultrafast Imaging (4 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 148 | Streak Camera Imaging | M --> Sigma --> D | M0 | Registered | [streak_camera.md](streak_camera.md) |
| 149 | Pump-Probe Microscopy | M --> R --> D | M0 | Registered | [pump_probe.md](pump_probe.md) |
| 150 | Compressed Ultrafast Photography (CUP) | M --> Sigma --> D | M0 | Registered | [cup.md](cup.md) |
| 151 | XFEL Serial Femtosecond Crystallography (SFX) | M --> R --> D | M0 | Registered | [xfel_sfx.md](xfel_sfx.md) |

---

## 16. Quantum Imaging (3 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 152 | Ghost Imaging | M --> Sigma --> D | M0 | Registered | [ghost_imaging.md](ghost_imaging.md) |
| 153 | Quantum Illumination | M --> R --> D | M0 | Planned | [quantum_illumination.md](quantum_illumination.md) |
| 154 | Entangled Photon Microscopy | M --> R --> D | M0 | Planned | [entangled_photon.md](entangled_photon.md) |

---

## 17. Multi-Modal Fusion (6 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 155 | PET/CT Fusion | Pi --> D (CT) + Pi --> D (PET) --> Fusion | M0 | Registered | [pet_ct.md](pet_ct.md) |
| 156 | PET/MR Fusion | Pi --> D (PET) + M --> F --> S --> D (MR) --> Fusion | M0 | Registered | [pet_mr.md](pet_mr.md) |
| 157 | SPECT/CT Fusion | Pi --> D (SPECT) + Pi --> D (CT) --> Fusion | M0 | Registered | [spect_ct.md](spect_ct.md) |
| 158 | US/MRI Fusion | P --> D (US) + M --> F --> S --> D (MR) --> Fusion | M0 | Registered | [us_mri.md](us_mri.md) |
| 159 | CT + Fluorescence (FLIT) | Pi --> D (CT) + M --> R,P --> D (FLI) --> Fusion | M0 | Registered | [ct_fluorescence.md](ct_fluorescence.md) |
| 160 | Correlative Light-Electron Microscopy (CLEM) | C --> D (LM) + C --> D (EM) --> Fusion | M0 | Registered | [clem.md](clem.md) |

---

## 18. Scanning Probe Microscopy (4 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 161 | Atomic Force Microscopy (AFM) | S --> D | M0 | Registered | [afm.md](afm.md) |
| 162 | Scanning Tunneling Microscopy (STM) | S --> D | M0 | Registered | [stm.md](stm.md) |
| 163 | Near-field Scanning Optical Microscopy (NSOM) | M --> C --> D | M0 | Registered | [nsom.md](nsom.md) |
| 164 | Magnetic Force Microscopy (MFM) | S --> M --> D | M0 | Registered | [mfm.md](mfm.md) |

---

## 19. Astronomy & Space Imaging (4 modalities)

| # | Modality | DAG | Maturity | Status | File |
|---|----------|-----|----------|--------|------|
| 165 | Stellar Coronagraphy | M --> P --> D | M0 | Registered | [coronagraphy.md](coronagraphy.md) |
| 166 | Lucky Imaging | M --> C --> D | M0 | Registered | [lucky_imaging.md](lucky_imaging.md) |
| 167 | Event Horizon Telescope (EHT) Imaging | F --> S --> D | M0 | Registered | [eht_imaging.md](eht_imaging.md) |
| 168 | Solar EUV/X-ray Imaging | M --> P --> D | M0 | Registered | [solar_imaging.md](solar_imaging.md) |

---

## References

- [Detailed Benchmark Specifications](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Imaging Modality Registry](../imaging_modalities.md)
