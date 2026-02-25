# Primitives Operators Database — 10 Primitives × 168 Imaging Modalities

## Summary

The Physics World Model uses exactly **10 canonical primitives** (plus an implicit **Source**) as a finite basis to express every imaging modality. Each primitive is a physics building block; modality-specific behavior comes from the **operator** (parameter) inside it. This database catalogues all operators across all **168 modalities** in **19 categories**.

| Stat | Value |
|------|-------|
| **Canonical primitives** | 10 (+1 implicit Source) |
| **Total modalities** | 168 |
| **Categories** | 19 |
| **Unique DAG topologies** | 12 families |
| **B2 mismatch parameters** | ~794 |

---

## The 10 Canonical Primitives

| # | Primitive | Symbol | Meaning | Used In | Description |
|---|-----------|:------:|---------|:-------:|-------------|
| 1 | **Detect** | `D` | Detector (gain + noise) | 168 | Universal terminal node — every spec ends with D |
| 2 | **Modulate** | `M` | Mask / spatial / temporal modulation | 98 | Binary masks, coded apertures, phase modulators, polarizers, illumination patterns |
| 3 | **Convolve** | `C` | Convolution / PSF / blur | 52 | Point spread functions, contrast transfer functions, optical blur |
| 4 | **Propagate** | `P` | Wave / field propagation | 38 | Acoustic, EM, electron, Fresnel, diffuse propagation |
| 5 | **Project** | `Π` | Tomographic / angular projection | 32 | Fan-beam, cone-beam, parallel, line-of-response projections |
| 6 | **Sample** | `S` | Spatial / spectral sampling | 28 | Raster scanning, spectral binning, probe scanning |
| 7 | **Scatter** | `R` | Elastic / inelastic scattering | 24 | Raman, Rayleigh, fluorescence, X-ray fluorescence, SHG |
| 8 | **Encode** | `F` | Fourier / frequency encoding | 20 | k-space sampling, synthetic aperture, diffraction patterns |
| 9 | **Accumulate** | `Σ` | Signal integration / compression | 16 | Temporal sum, spectral sum, interferometric sum, volume rendering |
| 10 | **Disperse** | `W` | Spectral / wavelength separation | 6 | Prism, grating, energy-dispersive separation |

> **Source (`Src`)** is implicit in all modalities — it specifies the carrier type (photon, electron, acoustic, RF, etc.) but is not counted as a separate primitive in the DAG.

---

## DAG Family Classification

| DAG Family | Primitive Chain | Count | Representative Modalities |
|------------|----------------|:-----:|---------------------------|
| Deconvolution | `C → D` | 32 | widefield, SEM, TEM, fundus, lensless, cryo_em |
| Tomography | `Π → D` | 24 | CT, PET, SPECT, mammography, neutron_tomo |
| MRI family | `M → F → S → D` | 12 | MRI, fMRI, MRS, diffusion_mri, ASL, MR fingerprinting |
| Ptychographic | `M → P → D` | 10 | ptychography, FPM, coronagraphy, shearography |
| Propagation | `P → D` | 22 | ultrasound, sonar, holography, ToF, seismic |
| Scattering | `M → R → D` | 16 | Raman, CARS, FLIM, SHG, XRF imaging, Brillouin |
| Compressive | `M → Σ → D` | 8 | SPC, CACTI, ghost imaging, HDR, FTIR |
| Interferometric | `F → S → D` | 8 | radio interferometry, X-ray crystallography, EHT |
| Spectral | `M → W → Σ → D` | 5 | CASSI, hyperspectral remote sensing |
| Fourier | `F → D` | 3 | SAR, eddy current |
| Scanning probe | `S → D` | 8 | AFM, STM, MALDI-MSI, atom probe, EELS |
| Multi-modal | Combined DAGs → Fusion | 6 | PET/CT, PET/MR, CLEM, US/MRI |
| Other | Various | 14 | coded exposure, event camera, weather radar |

---

## Complete Operator Catalogue by Primitive

### D — Detect (168 modalities — universal)

Every modality ends with a detector. The operator specifies gain `g`, noise level `η`, and detector type.

| Detector Class | Carrier | Example Detector Types | Modalities |
|----------------|---------|----------------------|:----------:|
| CCD/CMOS | Photon | Scientific CMOS, CCD, flat-panel | ~60 |
| Photon-counting | Photon | PMT, EMCCD, sCMOS, SPAD, TCSPC, APD | ~25 |
| Piezo/Transducer | Acoustic | Phased array, hydrophone, transducer | ~12 |
| RF Coil | Spin/RF | Surface coil, phased-array coil | ~12 |
| Electron detector | Electron | Direct electron detector, scintillator | ~11 |
| Scintillation | Gamma | Gamma camera, PET ring | ~8 |
| Ion detector | Ion | TOF detector, channeltron | ~4 |
| Coded aperture | Photon | Coded-aperture focal plane | ~2 |
| Thermal sensor | Thermal/THz | Bolometer, microbolometer | ~2 |
| Mechanical | Mechanical | Cantilever deflection | ~2 |
| Magnetic | Magnetic | SQUID, pickup coil | ~2 |
| Bucket/single-pixel | Photon | Single-pixel detector | ~3 |

---

### M — Modulate (98 modalities)

| Operator | Label | Modalities Using |
|----------|-------|-----------------|
| `mask` | Binary coded aperture | cassi, sd_cassi |
| `Φ` | Sensing matrix | spc, spc_block, matrix |
| `H⊗W` | Kronecker sensing | spc_kronecker |
| `m_t` | Temporal mask | cacti |
| `polarizer` | Polarizer / analyzer | polarization |
| `grating` | Sinusoidal grating pattern | sim |
| `LED_array` | LED array illumination | fpm |
| `block` | Block illumination | spc_block |
| `pattern` | Projected pattern | structured_light |
| `phase` | Phase modulation | phase_contrast, dic |
| `exposure_code` | Temporal exposure code | coded_exposure |
| `threshold` | Event threshold | event_camera |
| `exposure_bracket` | Multi-exposure bracket | hdr_imaging |
| `DMD` | Digital micromirror device | cup, ghost_imaging |
| `stochastic_activation` | Single-molecule activation | palm_storm, dna_paint |
| `pump` | Pump beam excitation | flim, shg, pump_probe, cars, srs, libs, brillouin |
| `entangled_source` | Entangled photon source | quantum_illumination, entangled_photon |
| `saturation` | Saturation/depletion beam | sted (implicit in C) |
| `coronagraph_mask` | Coronagraph stop | coronagraphy |
| `frame_selection` | Lucky frame selection | lucky_imaging |
| `modulated_source` | Modulated RF/current | mri, fmri, mrs, diffusion_mri, mr_elastography, cest_mri, asl_mri, mra, swi, mr_fingerprinting |
| `fiber_coupling` | Fiber-optic coupling | endoscopy, confocal_endomicroscopy, nirs_brain |
| `deformable_mirror` | Wavefront modulator | adaptive_optics |
| `electrode_pattern` | Electrode drive pattern | impedance_tomo |
| `excitation` | Excitation illumination | xrf_imaging, edx_mapping, cathodoluminescence |
| `FFP` | Focus field pattern | magnetic_particle |
| `spectral_filter` | Band-pass / filter wheel | multispectral_sat, ocean_color, ftir_imaging, streak_camera |
| `solar_filter` | Solar wavelength filter | solar_imaging |
| `aperture_coding` | Aperture function | nerf, gaussian_splatting |
| `scan_pattern` | Raster/vector scan | electron_diffraction |
| `diffuser_mask` | Diffuser/scattering mask | photometric_stereo |
| `polSAR` | Polarization modulation | polsar |
| `nsom_probe` | Near-field probe coupling | nsom |
| `shear` | Shearing optic | shearography |

---

### C — Convolve (52 modalities)

| Operator | Label | Modalities Using |
|----------|-------|-----------------|
| `PSF` | Widefield PSF | widefield, widefield_lowdose, sim, flim, polarization |
| `PSF_confocal` | Confocal PSF | confocal_livecell |
| `PSF_3D` | 3D confocal PSF | confocal_3d |
| `PSF_sheet` | Light-sheet PSF | lightsheet |
| `PSF_2P` | Two-photon PSF | two_photon |
| `PSF_3P` | Three-photon PSF | three_photon |
| `PSF_STED` | STED effective PSF | sted |
| `PSF_TIRF` | Evanescent-field PSF | tirf |
| `PSF_optic` | Ophthalmic PSF | fundus |
| `PSF_fiber` | Fiber bundle PSF | endoscopy, confocal_endomicroscopy |
| `PSF_focus` | Depth-dependent PSF | panorama |
| `PSF_expand` | Expansion-scaled PSF | expansion |
| `PSF_minflux` | Donut/zero-intensity PSF | minflux |
| `PSF_ISM` | ISM detector-array PSF | ism |
| `PSF_lattice` | Lattice light-sheet PSF | lattice_lightsheet |
| `PSF_spinning` | Spinning-disk PSF | spinning_disk |
| `PSF_dark` | Dark-field condenser PSF | dark_field |
| `PSF_phase` | Phase contrast PSF | phase_contrast |
| `PSF_DIC` | DIC shear PSF | dic |
| `CTF` | Contrast transfer function | tem, cryo_em |
| `probe` | Electron probe function | sem, stem |
| `PSF_AO` | AO-corrected PSF | adaptive_optics |
| `PSF_lucky` | Turbulence-selected PSF | lucky_imaging |
| `PSF_nsom` | Near-field PSF | nsom |
| `PSF_motion` | Motion blur PSF | coded_exposure |
| `PSF_structured` | Structured-light blur | structured_light, photometric_stereo |
| `PSF_machine` | Machine vision optics | machine_vision |
| `PSF_LM` | Light microscope PSF | clem (LM arm) |
| `PSF_EM` | Electron microscope PSF | clem (EM arm) |
| `PSF_FIB` | FIB-SEM imaging PSF | fib_sem |
| `PSF_SAM` | Acoustic lens PSF | acoustic_microscopy (implicit in P) |
| `PSF_micro` | Microlens array PSF | light_field, integral |

---

### P — Propagate (38 modalities)

| Operator | Label | Modalities Using |
|----------|-------|-----------------|
| `acoustic` | Acoustic wave propagation | ultrasound, doppler_ultrasound, photoacoustic, sonar, elastography, ivus, ceus, ultrasonic_phased_array, acoustic_microscopy, acoustic_emission |
| `e⁻` | Electron beam propagation | tem, sem, stem, electron_holography, electron_diffraction, electron_tomography |
| `Fresnel` | Fresnel diffraction | holography |
| `far-field` | Fraunhofer diffraction | phase_retrieval |
| `diffuse` | Diffuse photon propagation | dot, nirs_brain, bioluminescence_tomo |
| `low-coherence` | Low-coherence interferometric | oct, octa |
| `modulated` | Modulated CW propagation | tof_camera |
| `pulsed` | Pulsed laser propagation | lidar, flash_lidar |
| `probe` | Ptychographic probe illumination | ptychography, fpm |
| `shear` | Shear-wave propagation | elastography (implicit) |
| `diffuser` | Diffuser propagation | lensless |
| `RF` | RF/EM wave propagation | gpr, weather_radar |
| `THz` | Terahertz propagation | terahertz |
| `thermal` | Thermal diffusion | active_thermography |
| `seismic` | Seismic wave propagation | seismic_tomo, fwi, ocean_acoustic_tomo |
| `GW` | Gravitational wave propagation | gravitational_wave |
| `coherent_scatter` | Coherent scattering propagation | odt, talbot_lau |
| `solar` | Solar photon propagation | solar_imaging |
| `coronagraph` | Coronagraphic propagation | coronagraphy |
| `nerf_ray` | Neural radiance ray marching | nerf |
| `splat` | Gaussian splatting ray | gaussian_splatting |
| `shearography` | Shearographic interferometric | shearography |

---

### Π — Project (32 modalities)

| Operator | Label | Modalities Using |
|----------|-------|-----------------|
| `fan` | Fan-beam projection | ct, industrial_ct |
| `cone` | Cone-beam projection | cbct |
| `proj` | Generic X-ray projection | xray_radiography, angiography, fluoroscopy, dexa, xray_ndt, portal_imaging |
| `contact` | Contact/compression projection | mammography, digital_breast_tomo |
| `parallel` | Parallel-hole collimator | spect, spect_ct (SPECT arm) |
| `LOR` | Line-of-response (coincidence) | pet, pet_ct (PET arm), pet_mr (PET arm) |
| `neutron` | Neutron attenuation | neutron_tomo |
| `proton` | Proton transmission | proton_radiography, proton_therapy_img |
| `muon` | Muon scattering | muon_tomo |
| `electron_tilt` | Electron tilt-series projection | electron_tomography, cryo_et |
| `ray` | Volume ray casting | nerf |
| `splat` | Gaussian splatting | gaussian_splatting |
| `brachytherapy` | Brachytherapy dose projection | brachytherapy_img |
| `spectral_ct` | Energy-resolved projection | spectral_ct |
| `xrf_proj` | XRF sinogram projection | xrf_tomo |
| `CT_fused` | CT projection (fusion) | pet_ct, spect_ct, ct_fluorescence |

---

### S — Sample (28 modalities)

| Operator | Label | Modalities Using |
|----------|-------|-----------------|
| `raster` | Raster scan | stem, fib_sem |
| `spectral_bin` | Spectral binning / energy window | eels, sims, desi, maldi_msi |
| `k-space` | k-space trajectory sampling | mri, fmri, mrs, diffusion_mri (via F→S) |
| `field_point` | Atom-by-atom field evaporation | atom_probe |
| `LiDAR_scan` | LiDAR scanning pattern | lidar |
| `acoustic_scan` | Acoustic emission array | acoustic_emission |
| `visibility` | Visibility sampling (interferometric) | radio_interferometry, insar, radio_astronomy, eht_imaging |
| `diffraction_spot` | Diffraction spot integration | xray_crystallography, neutron_diffraction |
| `cantilever` | Cantilever raster scan | afm |
| `tunnel` | Tunneling current scan | stm |

---

### R — Scatter (24 modalities)

| Operator | Label | Modalities Using |
|----------|-------|-----------------|
| `Raman` | Spontaneous Raman scattering | raman_imaging |
| `CARS` | Coherent anti-Stokes Raman | cars |
| `SRS` | Stimulated Raman scattering | srs |
| `Brillouin` | Brillouin scattering | brillouin |
| `SHG` | Second harmonic generation | shg |
| `LIBS` | Laser-induced breakdown | libs |
| `XRF` | X-ray fluorescence | xrf_imaging, xrf_tomo |
| `CL` | Cathodoluminescence | cathodoluminescence |
| `EDX` | Energy-dispersive X-ray | edx_mapping |
| `fluorescence_lifetime` | Fluorescence with lifetime | flim |
| `pump_probe` | Pump-probe transient | pump_probe |
| `XFEL` | XFEL diffraction/damage | xfel_sfx |
| `quantum` | Quantum correlation scattering | quantum_illumination, entangled_photon |
| `Kikuchi` | Kikuchi/EBSD pattern | ebsd |
| `SAXS` | Small-angle X-ray scattering | saxs |
| `WAXS` | Wide-angle X-ray scattering | waxs |
| `calorimetry` | Particle shower | particle_calorimetry |
| `bubble_harmonic` | Microbubble nonlinear | ceus |
| `weather` | Hydrometeor scattering | weather_radar |
| `bioluminescence` | Bioluminescent emission | bioluminescence_tomo |
| `optical_diffuse` | Diffuse optical scattering | dot, nirs_brain |

---

### F — Encode (20 modalities)

| Operator | Label | Modalities Using |
|----------|-------|-----------------|
| `k-traj` | Cartesian k-space | mri |
| `EPI` | Echo-planar encoding | fmri, diffusion_mri |
| `FID` | Free induction decay | mrs |
| `MRE_MEG` | Motion-encoding gradient | mr_elastography |
| `CEST_sat` | Chemical exchange saturation | cest_mri |
| `ASL_label` | Arterial spin labeling | asl_mri |
| `MRA_encode` | Angiographic encoding | mra |
| `SWI_phase` | Susceptibility-weighted phase | swi |
| `MRF_schedule` | Fingerprinting sequence | mr_fingerprinting |
| `azimuth×range` | Range-Doppler encoding | sar |
| `InSAR` | Interferometric SAR phase | insar |
| `polSAR_encode` | Polarimetric SAR encoding | polsar |
| `diffraction` | Electron diffraction pattern | electron_diffraction |
| `eddy_current` | Eddy current impedance | eddy_current |
| `MPI_FFP` | MPI frequency encoding | magnetic_particle |
| `visibility` | Radio visibility encoding | radio_interferometry, radio_astronomy, eht_imaging |
| `crystal_diffraction` | Bragg diffraction | xray_crystallography |
| `neutron_TOF` | Neutron time-of-flight | neutron_diffraction |

---

### Σ — Accumulate (16 modalities)

| Operator | Label | Modalities Using |
|----------|-------|-----------------|
| `t` | Temporal integration | ultrasound, doppler_ultrasound, photoacoustic, pet, fluoroscopy, elastography, cacti, fmri, sonar, flim |
| `λ` | Spectral sum | cassi, sd_cassi, hyperspectral_remote |
| `θ` | Angular sum | fpm |
| `φ` | Phase-shift sum | sim |
| `f` | Focus stack sum | panorama |
| `E` | Energy window sum | spect |
| `interference` | Interferometric sum | oct, octa, electron_holography |
| `volume` | Volume rendering | nerf |
| `alpha` | Alpha compositing | gaussian_splatting |
| `correlation` | Correlation integration | tof_camera |
| `return` | Return signal integration | lidar |
| `GW_integration` | Gravitational wave integration | gravitational_wave |
| `calorimetry` | Shower energy integration | particle_calorimetry |
| `ghost` | Ghost imaging correlation sum | ghost_imaging |
| `CUP` | Compressed ultrafast sum | cup |
| `HDR` | Multi-exposure merge | hdr_imaging |
| `FTIR` | Interferogram integration | ftir_imaging |
| `streak` | Streak camera temporal sum | streak_camera |
| `passive` | Passive radiometric integration | passive_microwave |
| `multispectral` | Multi-band sum | multispectral_sat, ocean_color |

---

### W — Disperse (6 modalities)

| Operator | Label | Modalities Using |
|----------|-------|-----------------|
| `prism` | Prism dispersion | cassi, sd_cassi |
| `grating` | Diffraction grating | hyperspectral_remote |
| `energy_bin` | Energy-dispersive binning | spectral_ct |
| `EELS_disperser` | Electron energy loss disperser | eels (implicit in S) |
| `spectrometer` | Spectrometer dispersion | raman_imaging, libs, brillouin (implicit in R) |

---

## All 168 Modality Specs

### Category 1: Microscopy (24)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 1 | `widefield` | C → D | Photon | M3 |
| 2 | `widefield_lowdose` | C → D | Photon | M1 |
| 3 | `confocal_livecell` | C → D | Photon | M1 |
| 4 | `confocal_3d` | C → D | Photon | M1 |
| 5 | `sim` | M → C → D | Photon | M2 |
| 6 | `lightsheet` | C → D | Photon | M1 |
| 7 | `flim` | M → R → D | Photon | M0 |
| 8 | `fpm` | M → P → D | Photon | M1 |
| 9 | `two_photon` | C → D | Photon | M0 |
| 10 | `sted` | C → D | Photon | M0 |
| 11 | `palm_storm` | M → D | Photon | M0 |
| 12 | `tirf` | C → D | Photon | M0 |
| 13 | `polarization` | M → C → D | Photon | M0 |
| 14 | `expansion` | C → D | Photon | M0 |
| 15 | `minflux` | C → D | Photon | M0 |
| 16 | `ism` | C → D | Photon | M0 |
| 17 | `phase_contrast` | C → D | Photon | M0 |
| 18 | `dic` | M → C → D | Photon | M0 |
| 19 | `dark_field` | C → D | Photon | M0 |
| 20 | `lattice_lightsheet` | C → D | Photon | M0 |
| 21 | `shg` | M → R → D | Photon | M0 |
| 22 | `spinning_disk` | C → D | Photon | M0 |
| 23 | `three_photon` | C → D | Photon | M0 |
| 24 | `dna_paint` | M → D | Photon | M0 |

### Category 2: Compressive Imaging (4)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 25 | `cassi` | M → W → Σ → D | Photon | M3 |
| 26 | `spc` | M → Σ → D | Photon | M3 |
| 27 | `cacti` | M → Σ → D | Photon | M3 |
| 28 | `matrix` | M → D | Varies | M1 |

### Category 3: Medical Imaging (37)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 29 | `ct` | Π → D | X-ray | M3 |
| 30 | `mri` | M → F → S → D | Spin/RF | M3 |
| 31 | `xray_radiography` | Π → D | X-ray | M0 |
| 32 | `ultrasound` | P → D | Acoustic | M1 |
| 33 | `pet` | Π → D | Gamma | M1 |
| 34 | `spect` | Π → D | Gamma | M0 |
| 35 | `fluoroscopy` | Π → D | X-ray | M0 |
| 36 | `mammography` | Π → D | X-ray | M0 |
| 37 | `dexa` | Π → D | X-ray | M0 |
| 38 | `cbct` | Π → D | X-ray | M0 |
| 39 | `angiography` | Π → D | X-ray | M0 |
| 40 | `dot` | M → R,P,R → D | Photon | M0 |
| 41 | `photoacoustic` | M → P → D | Acoustic | M0 |
| 42 | `oct` | P+P → Σ → D | Photon | M1 |
| 43 | `fmri` | M → F → S → D | Spin/RF | M0 |
| 44 | `mrs` | M → F → S → D | Spin/RF | M0 |
| 45 | `diffusion_mri` | M → F → S → D | Spin/RF | M0 |
| 46 | `doppler_ultrasound` | P → D | Acoustic | M0 |
| 47 | `elastography` | P → D | Acoustic | M0 |
| 48 | `endoscopy` | M → C → D | Photon | M0 |
| 49 | `fundus` | C → D | Photon | M0 |
| 50 | `octa` | P+P → Σ → D | Photon | M0 |
| 51 | `proton_therapy_img` | Π → D | Proton | M0 |
| 52 | `brachytherapy_img` | Π → D | Gamma | M0 |
| 53 | `portal_imaging` | Π → D | MV X-ray | M0 |
| 54 | `spectral_ct` | Π → W → D | X-ray | M0 |
| 55 | `mr_elastography` | M → F → S → D | Spin/RF | M0 |
| 56 | `cest_mri` | M → F → S → D | Spin/RF | M0 |
| 57 | `asl_mri` | M → F → S → D | Spin/RF | M0 |
| 58 | `mra` | M → F → S → D | Spin/RF | M0 |
| 59 | `swi` | M → F → S → D | Spin/RF | M0 |
| 60 | `mr_fingerprinting` | M → F → S → D | Spin/RF | M0 |
| 61 | `ivus` | P → D | Acoustic | M0 |
| 62 | `ceus` | P → R → D | Acoustic | M0 |
| 63 | `digital_breast_tomo` | Π → D | X-ray | M0 |
| 64 | `confocal_endomicroscopy` | M → C → D | Photon | M0 |
| 65 | `nirs_brain` | M → R,P → D | Photon | M0 |

### Category 4: Coherent Imaging (5)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 66 | `ptychography` | M → P → D | Electron/Photon | M3 |
| 67 | `holography` | P → D | Photon | M1 |
| 68 | `phase_retrieval` | P → D | Photon/Electron | M0 |
| 69 | `odt` | P → D | Photon | M0 |
| 70 | `talbot_lau` | M → P → D | X-ray | M0 |

### Category 5: Computational Photography (5)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 71 | `lensless` | C → D | Photon | M3 |
| 72 | `panorama` | C → D | Photon | M0 |
| 73 | `coded_exposure` | M → C → D | Photon | M0 |
| 74 | `event_camera` | M → D | Photon | M0 |
| 75 | `hdr_imaging` | M → Σ → D | Photon | M0 |

### Category 6: Computational Optics (2)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 76 | `light_field` | C → S → D | Photon | M0 |
| 77 | `integral` | C → S → D | Photon | M0 |

### Category 7: Neural Rendering (2)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 78 | `nerf` | M → P → D | Photon | M0 |
| 79 | `gaussian_splatting` | M → P → D | Photon | M0 |

### Category 8: Electron Microscopy (11)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 80 | `sem` | C → D | Electron | M0 |
| 81 | `tem` | C → D | Electron | M0 |
| 82 | `electron_tomography` | Π → D | Electron | M0 |
| 83 | `stem` | S → D | Electron | M0 |
| 84 | `electron_diffraction` | M → P → D | Electron | M0 |
| 85 | `ebsd` | R → D | Electron | M0 |
| 86 | `eels` | S → D | Electron | M0 |
| 87 | `electron_holography` | P → D | Electron | M0 |
| 88 | `cryo_et` | Π → D | Electron | M0 |
| 89 | `fib_sem` | S → C → D | Electron+Ion | M0 |
| 90 | `edx_mapping` | M → R → D | Electron | M0 |

### Category 9: Depth Imaging (5)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 91 | `tof_camera` | P → D | Photon | M0 |
| 92 | `lidar` | P → S → D | Photon | M0 |
| 93 | `structured_light` | M → C → D | Photon | M0 |
| 94 | `photometric_stereo` | M → C → D | Photon | M0 |
| 95 | `flash_lidar` | P → D | Photon | M0 |

### Category 10: Remote Sensing (11)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 96 | `sar` | F → D | RF | M0 |
| 97 | `sonar` | P → D | Acoustic | M0 |
| 98 | `hyperspectral_remote` | M → W → Σ → D | Photon | M0 |
| 99 | `multispectral_sat` | M → Σ → D | Photon | M0 |
| 100 | `gpr` | P → D | RF | M0 |
| 101 | `weather_radar` | P → R → D | RF | M0 |
| 102 | `radio_interferometry` | F → S → D | RF | M0 |
| 103 | `passive_microwave` | Σ → D | RF | M0 |
| 104 | `insar` | F → S → D | RF | M0 |
| 105 | `polsar` | F → M → D | RF | M0 |
| 106 | `ocean_color` | M → Σ → D | Photon | M0 |

### Category 11: Industrial Inspection (10)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 107 | `industrial_ct` | Π → D | X-ray | M0 |
| 108 | `xray_ndt` | Π → D | X-ray | M0 |
| 109 | `ultrasonic_phased_array` | P → D | Acoustic | M0 |
| 110 | `eddy_current` | F → D | RF | M0 |
| 111 | `active_thermography` | P → D | Thermal | M0 |
| 112 | `terahertz` | P → D | THz | M0 |
| 113 | `machine_vision` | C → D | Photon | M0 |
| 114 | `xrf_imaging` | M → R → D | Photon | M0 |
| 115 | `shearography` | M → P → D | Photon | M0 |
| 116 | `acoustic_microscopy` | P → D | Acoustic | M0 |

### Category 12: Scientific Instrumentation (12)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 117 | `xray_crystallography` | F → S → D | X-ray | M0 |
| 118 | `saxs` | R → D | X-ray | M0 |
| 119 | `maldi_msi` | S → D | Ion | M0 |
| 120 | `atom_probe` | S → D | Ion | M0 |
| 121 | `cryo_em` | C → D | Electron | M3 |
| 122 | `neutron_tomo` | Π → D | Neutron | M0 |
| 123 | `proton_radiography` | Π → D | Proton | M0 |
| 124 | `muon_tomo` | Π → D | Muon | M0 |
| 125 | `waxs` | R → D | X-ray | M0 |
| 126 | `xrf_tomo` | Π → R → D | X-ray | M0 |
| 127 | `neutron_diffraction` | R → S → D | Neutron | M0 |
| 128 | `cathodoluminescence` | M → R → D | Electron | M0 |

### Category 13: Broader Experimental Science (11)

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 129 | `adaptive_optics` | M → C → D | Photon | M0 |
| 130 | `seismic_tomo` | P → D | Acoustic | M0 |
| 131 | `gravitational_wave` | P → Σ → D | GW | M0 |
| 132 | `particle_calorimetry` | R → Σ → D | Photon | M0 |
| 133 | `radio_astronomy` | F → S → D | RF | M0 |
| 134 | `acoustic_emission` | P → S → D | Acoustic | M0 |
| 135 | `magnetic_particle` | M → F → D | Magnetic | M0 |
| 136 | `impedance_tomo` | M → D | Electrical | M0 |
| 137 | `fwi` | P → D | Seismic/Acoustic | M0 |
| 138 | `ocean_acoustic_tomo` | P → D | Acoustic | M0 |
| 139 | `bioluminescence_tomo` | Src → R,P → D | Photon | M0 |

### Category 14: Spectroscopy & Spectral Imaging (8) — NEW

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 140 | `raman_imaging` | M → R → D | Photon | M0 |
| 141 | `cars` | M → R → D | Photon | M0 |
| 142 | `srs` | M → R → D | Photon | M0 |
| 143 | `ftir_imaging` | M → Σ → D | IR photon | M0 |
| 144 | `libs` | M → R → D | Photon | M0 |
| 145 | `brillouin` | M → R → D | Photon | M0 |
| 146 | `sims` | S → D | Ion | M0 |
| 147 | `desi` | S → D | Ion | M0 |

### Category 15: Ultrafast Imaging (4) — NEW

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 148 | `streak_camera` | M → Σ → D | Photon | M0 |
| 149 | `pump_probe` | M → R → D | Photon | M0 |
| 150 | `cup` | M → Σ → D | Photon | M0 |
| 151 | `xfel_sfx` | M → R → D | X-ray | M0 |

### Category 16: Quantum Imaging (3) — NEW

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 152 | `ghost_imaging` | M → Σ → D | Photon | M0 |
| 153 | `quantum_illumination` | M → R → D | Photon | M0 |
| 154 | `entangled_photon` | M → R → D | Photon | M0 |

### Category 17: Multi-Modal Fusion (6) — NEW

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 155 | `pet_ct` | Π→D + Π→D → Fusion | X-ray + Gamma | M0 |
| 156 | `pet_mr` | Π→D + M→F→S→D → Fusion | Gamma + RF | M0 |
| 157 | `spect_ct` | Π→D + Π→D → Fusion | Gamma + X-ray | M0 |
| 158 | `us_mri` | P→D + M→F→S→D → Fusion | Acoustic + RF | M0 |
| 159 | `ct_fluorescence` | Π→D + M→R,P→D → Fusion | X-ray + Photon | M0 |
| 160 | `clem` | C→D + C→D → Fusion | Photon + Electron | M0 |

### Category 18: Scanning Probe Microscopy (4) — NEW

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 161 | `afm` | S → D | Mechanical | M0 |
| 162 | `stm` | S → D | Electron | M0 |
| 163 | `nsom` | M → C → D | Photon | M0 |
| 164 | `mfm` | S → M → D | Magnetic | M0 |

### Category 19: Astronomy & Space Imaging (4) — NEW

| # | Variant | DAG | Carrier | Maturity |
|---|---------|-----|---------|:--------:|
| 165 | `coronagraphy` | M → P → D | Photon | M0 |
| 166 | `lucky_imaging` | M → C → D | Photon | M0 |
| 167 | `eht_imaging` | F → S → D | RF (mm-wave) | M0 |
| 168 | `solar_imaging` | M → P → D | Photon/EUV | M0 |

---

## Carrier Types

| Carrier | Modalities |
|---------|:----------:|
| Photon (visible/IR/UV) | 78 |
| Electron | 16 |
| X-ray | 18 |
| Acoustic | 14 |
| Spin/RF | 12 |
| Gamma | 5 |
| RF (radar/radio) | 9 |
| Ion | 4 |
| Neutron | 2 |
| Proton | 2 |
| Muon | 1 |
| Thermal/THz | 2 |
| Mechanical | 1 |
| Magnetic | 2 |
| Electrical | 1 |
| Gravitational wave | 1 |
| EUV | 1 |

---

## Maturity Distribution

| Level | Description | Count |
|-------|-------------|:-----:|
| **M0** | Template only | 148 |
| **M1** | Synthetic data validated | 10 |
| **M2** | Compound mismatch tested | 1 |
| **M3** | Real experimental data | 7 |
| **M4** | Adversarial benchmarks | 0 |
| | **Total** | **168** |

### M3 Validated Modalities

| Variant | Validated Solver | PSNR Gain | ρ |
|---------|-----------------|-----------|---|
| `widefield` | Richardson-Lucy | +1 to +5 dB | ≥ 0.85 |
| `cassi` | GAP-TV | +0.76 dB | 0.85 |
| `spc` | FISTA-TV | +7.71 dB | 0.86 |
| `cacti` | GAP-TV | +10.21 dB | 1.00 |
| `ct` | FBP | +10.68 dB | 1.00 |
| `ptychography` | ePIE | +7.09 dB | 1.00 |
| `lensless` | ADMM | +3.55 dB | 0.78 |

---

## Key Observations

1. **Every modality is a chain of 2–4 primitives** (plus optional Fusion for multi-modal). The shortest specs have 2 nodes (e.g., `C → D`); the longest single-modality specs have 4 (e.g., `M → F → S → D`).

2. **D is universal** — every spec terminates with a detector. The carrier and noise model differentiate detector types.

3. **M is the most-used non-terminal primitive** (98 of 168 modalities). It encodes the system's active "question" to the scene — masks, patterns, pulse sequences, excitation beams.

4. **Operators carry the modality-specific physics.** The primitive `P` (Propagate) covers X-rays, electrons, acoustic waves, photons, seismic waves, and gravitational waves — the operator (`e⁻`, `acoustic`, `Fresnel`, `seismic`, `GW`) specifies which.

5. **The 10-primitive basis is complete.** All 168 modalities (spanning 17 carrier types) can be expressed without introducing new primitives. The 71 new modalities added beyond the original 97 required zero new primitive types.

6. **W (Disperse) is the least used primitive** (6 modalities). It could potentially merge with M (Modulate), but keeping it separate preserves the distinction between spatial modulation and spectral separation.

7. **Multi-modal fusion** introduces a new DAG pattern: two independent single-modality DAGs connected by a Fusion node. This is the only pattern requiring more than a linear chain.

8. **~794 total B2 mismatch parameters** are defined across all 168 modalities, providing a comprehensive space for robustness testing.
