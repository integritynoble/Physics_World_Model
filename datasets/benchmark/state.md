# Benchmark Dataset & Algorithm State Tracker

Last updated: 2026-03-11T21:00Z

## Status Legend

| State | Description |
|-------|-------------|
| **1) Dataset** | Creating public (>=10), dev (20), hidden (20) + spec.json + true_spec.json + per-sample images |
| **2) Benchmark** | Update https://pwm.platformai.org/benchmark -- gallery, data preview, baseline CPU recon |
| **3) Algorithms** | All algorithms tested (1 public sample each): classical CPU + GPU via Modal/server |
| **4) SpecLab** | All algorithm tests integrated into https://pwm.platformai.org/speclab |
| **5) Data Source** | Public dataset quality: `real` (popular published data) / `domain-sim` (domain-specific generator) / `synthetic` (universal generator) |
| **6) Quality Check** | Verified: no NaN/Inf/zeros, correct shapes, realistic value ranges, proper forward model keys |

Values: `done` / `in-progress` / `pending`

## Data Storage

All HDF5 datasets stored in GCS: `gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/{tier}/`

Download: `gsutil -m rsync -r gs://pwm-benchmark-datasets/datasets/Benchmark/ datasets/benchmark/`

---

## Priority 1 — Core Medical & Imaging

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | 5) Data Source | 6) Quality Check | Notes |
|----------|-----------|-------------|--------------|-----------|---------------|-----------------|-------|
| ct | done | done | 0/24 tested | pending | real | done | LoDoPaB-CT (Leuschner 2021, Sci Data 8:109); 362×362; 11 samples; fan-beam Radon |
| mri | done | done | 0/35 tested | pending | real | done | M4Raw (Lyu 2023, Sci Data); 256×256; 12 samples; multi-coil complex k-space |
| pet | done | done | 0/10 tested | pending | domain-sim | done | Zubal-like brain phantom; 256×256; 12 samples; Radon + Poisson |
| ultrasound | done | done | 0/14 tested | pending | domain-sim | done | PICMUS/CIRS-style phantom; 256×256; 12 samples; depth-dependent PSF + speckle |
| oct | done | done | 0/13 tested | pending | domain-sim | done | Retinal layer phantom; 256×256; 12 samples; axial PSF + speckle + rolloff |
| mammography | done | done | 0/13 tested | pending | domain-sim | done | Breast tissue phantom; 256×256; 12 samples; Beer-Lambert projection + Poisson |
| cbct | done | done | 0/9 tested | pending | domain-sim | done | Dental head phantom; 256×256; 12 samples; cone-beam geometry |
| spect | done | done | 0/10 tested | pending | domain-sim | done | Brain perfusion phantom; 256×256; 12 samples; depth-dependent CDR + Radon |
| fundus | done | done | 0/4 tested | pending | domain-sim | done | Fractal vessel tree phantom; 256×256; 12 samples; retinal PSF model |
| endoscopy | done | done | 0/9 tested | pending | domain-sim | done | Fiber PSF + LED illumination; 256×256; 12 samples |
| fmri | done | done | 0/35 tested | pending | domain-sim | done | Brain activation phantom; 256×256; 12 samples; hemodynamic forward model |
| diffusion_mri | done | done | 0/9 tested | pending | domain-sim | done | Fiber tract phantom; 256×256; 12 samples; diffusion tensor model |

## Priority 2 — Microscopy & Optical

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | 5) Data Source | 6) Quality Check | Notes |
|----------|-----------|-------------|--------------|-----------|---------------|-----------------|-------|
| palm_storm | done | done | 0/4 tested | pending | domain-sim | done | SMLM emitter phantom; 256×256; 12 samples; PSF + shot noise |
| sted | done | done | 0/13 tested | pending | domain-sim | done | Cell structure phantom; 256×256; 12 samples; STED depletion PSF |
| sim | done | done | 0/4 tested | pending | domain-sim | done | Fluorescence phantom; 256×256; 12 samples; structured illumination |
| confocal_3d | done | done | 0/9 tested | pending | domain-sim | done | 3D cell phantom; 256×256; 12 samples; confocal PSF |
| lightsheet | done | done | 0/13 tested | pending | domain-sim | done | Tissue clearing phantom; 256×256; 12 samples; lightsheet PSF |
| two_photon | done | done | 0/13 tested | pending | domain-sim | done | Neural tissue phantom; 256×256; 12 samples; 2P excitation PSF |
| cryo_em | done | done | 0/9 tested | pending | domain-sim | done | Protein complex phantom; 256×256; 12 samples; CTF + ice noise |
| sem | done | done | 0/4 tested | pending | domain-sim | done | Nanostructure phantom; 256×256; 12 samples; SEM imaging model |
| tem | done | done | 0/4 tested | pending | domain-sim | done | Crystal structure phantom; 256×256; 12 samples; TEM imaging model |
| widefield | done | done | 0/13 tested | pending | domain-sim | done | Cell sample phantom; 256×256; 12 samples; widefield PSF + noise |
| photoacoustic | done | done | 0/4 tested | pending | domain-sim | done | Vascular phantom; 256×256; 12 samples; wave propagation |

## Priority 3 — Computational & Advanced

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | 5) Data Source | 6) Quality Check | Notes |
|----------|-----------|-------------|--------------|-----------|---------------|-----------------|-------|
| holography | done | done | 0/14 tested | pending | domain-sim | done | Complex-field (amplitude+phase); 256×256; 12 samples; off-axis angular spectrum |
| ptychography | done | done | 0/4 tested | pending | domain-sim | done | Complex object phantom; 256×256; 12 samples; scanning diffraction |
| lensless | done | done | 0/4 tested | pending | domain-sim | done | Natural image phantom; 256×256; 12 samples; PSF convolution |
| gaussian_splatting | done | done | 0/11 tested | pending | domain-sim | done | Multi-view RGB; 256×256×3; 12 samples; 3DGS rendering model |
| phase_retrieval | done | done | 0/14 tested | pending | domain-sim | done | Complex-field phantom; 256×256; 12 samples; Fourier magnitude |
| fpm | done | done | 0/4 tested | pending | domain-sim | done | Complex object; 256×256; 12 samples; Fourier ptychography model |
| odt | done | done | 0/4 tested | pending | domain-sim | done | Refractive index phantom; 256×256; 12 samples; Born/Rytov diffraction |
| ghost_imaging | done | done | 0/10 tested | pending | domain-sim | done | Single-pixel; 256×256; 12 samples; compressed sensing model |
| nerf | done | done | 0/11 tested | pending | synthetic | done | NeRF rendering; 256×256; 12 samples; universal generator |

## Priority 4 — Spectroscopy & Remote Sensing

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | 5) Data Source | 6) Quality Check | Notes |
|----------|-----------|-------------|--------------|-----------|---------------|-----------------|-------|
| raman_imaging | done | done | 0/11 tested | pending | domain-sim | done | Spectral phantom; 256×256; 12 samples; Raman scattering model |
| ftir_imaging | done | done | 0/11 tested | pending | domain-sim | done | Spectral phantom; 256×256; 12 samples; FTIR absorption model |
| sar | done | done | 0/13 tested | pending | domain-sim | done | Terrain phantom; 256×256; 12 samples; SAR imaging model |
| lidar | done | done | 0/4 tested | pending | domain-sim | done | Terrain/urban phantom; 256×256; 12 samples; LiDAR point cloud |
| hyperspectral_remote | done | done | 0/4 tested | pending | synthetic | done | Universal generator; 256×256; 12 samples (note: not real Indian Pines) |
| insar | done | done | 0/4 tested | pending | domain-sim | done | Terrain deformation; 256×256; 12 samples; complex interferogram (y_real/y_imag) |
| multispectral_sat | done | done | 0/13 tested | pending | synthetic | done | Universal generator; 256×256; 12 samples |
| sd_cassi | done | done | 0/9 tested | pending | real | done | KAIST TSA (10 scenes, 256×256×28); coded aperture + disperser; 145 MB |
| cacti | done | done | 0/9 tested | pending | real | done | Real video data; 256×256×8; 20 samples; temporal coding |
| spc_kronecker | done | done | 0/4 tested | pending | domain-sim | done | Compressed sensing; 231×231; 11 samples; Kronecker sampling |

## Priority 5 — Nuclear & Particle / Multimodality

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | 5) Data Source | 6) Quality Check | Notes |
|----------|-----------|-------------|--------------|-----------|---------------|-----------------|-------|
| pet_ct | done | done | 0/13 tested | pending | domain-sim | done | Joint PET/CT (x_ct+x_pet, y_ct+y_pet); 256×256; 12 samples |
| pet_mr | done | done | 0/10 tested | pending | domain-sim | done | Joint PET/MR (x_mr+x_pet, y_mr+y_pet); 256×256; 12 samples; complex k-space |
| spect_ct | done | done | 0/4 tested | pending | domain-sim | done | SPECT/CT (x_spect+x_ct, y_spect+y_ct); 256×256; 12 samples |
| spectral_ct | done | done | 0/13 tested | pending | domain-sim | done | Dual-energy (y_low+y_high, x_water+x_bone+x_iodine); 256×256; 12 samples |
| industrial_ct | done | done | 0/4 tested | pending | domain-sim | done | Manufacturing phantom; 256×256; 12 samples; cone-beam CT |

## Priority 6 — Remaining Modalities (127 universal-generated)

All Priority 6 modalities use the universal generator with domain-appropriate phantom types and physics-based forward models (PSF/Radon/kspace/mask/identity). Data source: **synthetic**. All verified: 12 samples each, 256×256, no NaN/Inf/zeros.

| Modality | 1) Dataset | 2) Benchmark | 5) Data Source | 6) Quality Check | Forward Model | Notes |
|----------|-----------|-------------|---------------|-----------------|--------------|-------|
| acoustic_emission | done | done | synthetic | done | psf | Time-reversal imaging |
| acoustic_microscopy | done | done | synthetic | done | psf | Acoustic lens PSF |
| active_thermography | done | done | synthetic | done | psf | Thermal diffusion PSF |
| adaptive_optics | done | done | synthetic | done | psf | Atmospheric turbulence PSF |
| afm | done | done | synthetic | done | psf | Tip-sample convolution |
| angiography | done | done | synthetic | done | radon | X-ray projection |
| asl_mri | done | done | synthetic | done | kspace | k-space undersampling |
| atom_probe | done | done | synthetic | done | psf | Trajectory distortion |
| bioluminescence_tomo | done | done | synthetic | done | psf | Diffuse photon transport |
| brachytherapy_img | done | done | synthetic | done | radon | X-ray projection |
| brillouin | done | done | synthetic | done | identity | Spectral measurement |
| cars | done | done | synthetic | done | identity | Nonlinear spectral |
| cassi | pending | pending | pending | pending | - | |
| cathodoluminescence | done | done | synthetic | done | identity | CL emission |
| cest_mri | done | done | synthetic | done | kspace | CEST contrast |
| ceus | done | done | synthetic | done | identity | Bubble dynamics |
| clem | done | done | synthetic | done | identity | Correlative registration |
| coded_exposure | done | done | synthetic | done | psf | Flutter shutter PSF |
| confocal_endomicroscopy | done | done | synthetic | done | psf | Fiber bundle PSF |
| confocal_livecell | done | done | synthetic | done | psf | Confocal PSF |
| coronagraphy | done | done | synthetic | done | psf | Coronagraph PSF |
| cryo_et | done | done | synthetic | done | radon | Tilt-series projection |
| ct_fluorescence | done | done | synthetic | done | psf | XRF emission |
| cup | done | done | synthetic | done | mask | Compressed ultrafast |
| dark_field | done | done | synthetic | done | psf | Dark-field contrast |
| desi | done | done | synthetic | done | identity | Mass spec imaging |
| dexa | done | done | synthetic | done | radon | Dual-energy X-ray |
| dic | done | done | synthetic | done | psf | DIC gradient contrast |
| digital_breast_tomo | done | done | synthetic | done | radon | Limited-angle tomo |
| dna_paint | done | done | synthetic | done | psf | SMLM PSF |
| doppler_ultrasound | done | done | synthetic | done | identity | Doppler flow |
| dot | done | done | synthetic | done | psf | Diffuse optical |
| ebsd | done | done | synthetic | done | identity | Diffraction pattern |
| eddy_current | done | done | synthetic | done | psf | EC inspection |
| edx_mapping | done | done | synthetic | done | identity | X-ray spectral |
| eels | done | done | synthetic | done | identity | Energy loss spectral |
| eht_imaging | done | done | synthetic | done | kspace | Sparse UV sampling |
| elastography | done | done | synthetic | done | psf | Shear wave |
| electron_diffraction | done | done | synthetic | done | identity | Diffraction |
| electron_holography | done | done | synthetic | done | psf | Phase contrast |
| electron_tomography | done | done | synthetic | done | radon | Tilt-series |
| entangled_photon | done | done | synthetic | done | identity | Quantum correlation |
| event_camera | done | done | synthetic | done | identity | Event-based |
| expansion | done | done | synthetic | done | psf | ExM PSF |
| fib_sem | done | done | synthetic | done | psf | FIB-SEM imaging |
| flash_lidar | done | done | synthetic | done | psf | SPAD array |
| flim | done | done | synthetic | done | identity | Lifetime decay |
| fluoroscopy | done | done | synthetic | done | radon | X-ray projection |
| fwi | done | done | synthetic | done | psf | Seismic wave |
| gpr | done | done | synthetic | done | radon | GPR migration |
| gravitational_wave | done | done | synthetic | done | identity | GW strain signal |
| hdr_imaging | done | done | synthetic | done | psf | Tone mapping |
| impedance_tomo | done | done | synthetic | done | psf | EIT current injection |
| integral | done | done | synthetic | done | psf | Light field PSF |
| ism | done | done | synthetic | done | psf | ISM PSF |
| ivus | done | done | synthetic | done | psf | IVUS beam |
| lattice_lightsheet | done | done | synthetic | done | psf | Lattice PSF |
| libs | done | done | synthetic | done | identity | LIBS emission |
| light_field | done | done | synthetic | done | psf | Light field |
| lucky_imaging | done | done | synthetic | done | psf | Atmospheric PSF |
| machine_vision | done | done | synthetic | done | psf | Imaging PSF |
| magnetic_particle | done | done | synthetic | done | psf | MPI system function |
| maldi_msi | done | done | synthetic | done | identity | MALDI spectral |
| matrix | done | done | synthetic | done | mask | Matrix sensing |
| mfm | done | done | synthetic | done | psf | Magnetic tip PSF |
| minflux | done | done | synthetic | done | psf | MINFLUX PSF |
| mr_elastography | done | done | synthetic | done | kspace | MRE k-space |
| mr_fingerprinting | done | done | synthetic | done | kspace | MRF k-space |
| mra | done | done | synthetic | done | kspace | MRA k-space |
| mrs | done | done | synthetic | done | kspace | MRS k-space |
| multispectral_sat | done | done | synthetic | done | psf | Satellite PSF |
| muon_tomo | done | done | synthetic | done | radon | Muon scattering |
| neutron_diffraction | done | done | synthetic | done | identity | Diffraction |
| neutron_tomo | done | done | synthetic | done | radon | Neutron projection |
| nirs_brain | done | done | synthetic | done | psf | fNIRS sensitivity |
| nsom | done | done | synthetic | done | psf | Near-field tip PSF |
| ocean_acoustic_tomo | done | done | synthetic | done | psf | Ocean acoustic |
| ocean_color | done | done | synthetic | done | identity | Ocean reflectance |
| octa | done | done | synthetic | done | psf | OCT angiography |
| panorama | done | done | synthetic | done | psf | Panoramic stitching |
| particle_calorimetry | done | done | synthetic | done | identity | Calorimeter response |
| passive_microwave | done | done | synthetic | done | psf | MW radiometer |
| phase_contrast | done | done | synthetic | done | psf | Phase contrast TIE |
| photometric_stereo | done | done | synthetic | done | identity | Multi-light |
| polarization | done | done | synthetic | done | psf | Polarization PSF |
| polsar | done | done | synthetic | done | psf | PolSAR imaging |
| portal_imaging | done | done | synthetic | done | radon | EPID projection |
| proton_radiography | done | done | synthetic | done | radon | Proton projection |
| proton_therapy_img | done | done | synthetic | done | radon | Proton CT |
| pump_probe | done | done | synthetic | done | identity | Ultrafast spectral |
| quantum_illumination | done | done | synthetic | done | identity | Quantum sensing |
| radio_astronomy | done | done | synthetic | done | kspace | UV plane sampling |
| radio_interferometry | done | done | synthetic | done | kspace | Interferometric UV |
| saxs | done | done | synthetic | done | identity | SAXS pattern |
| seismic_tomo | done | done | synthetic | done | radon | Seismic raypath |
| shearography | done | done | synthetic | done | psf | Shear interferometry |
| shg | done | done | synthetic | done | psf | SHG microscopy PSF |
| sims | done | done | synthetic | done | identity | SIMS spectral |
| solar_imaging | done | done | synthetic | done | psf | Solar telescope PSF |
| sonar | done | done | synthetic | done | psf | Sonar beam |
| spc | pending | pending | pending | pending | - | |
| spc_block | done | done | synthetic | done | mask | SPC block sampling |
| spinning_disk | done | done | synthetic | done | psf | Spinning disk PSF |
| srs | done | done | synthetic | done | identity | SRS spectral |
| stem | done | done | synthetic | done | psf | STEM probe |
| stm | done | done | synthetic | done | psf | STM tip PSF |
| streak_camera | done | done | synthetic | done | mask | Streak temporal coding |
| structured_light | done | done | synthetic | done | psf | Fringe projection |
| swi | done | done | synthetic | done | kspace | SWI k-space |
| talbot_lau | done | done | synthetic | done | psf | Grating interferometry |
| terahertz | done | done | synthetic | done | psf | THz PSF |
| three_photon | done | done | synthetic | done | psf | 3P microscopy PSF |
| tirf | done | done | synthetic | done | psf | TIRF evanescent PSF |
| tof_camera | done | done | synthetic | done | psf | ToF multi-path |
| ultrasonic_phased_array | done | done | synthetic | done | psf | Phased array beam |
| us_mri | done | done | synthetic | done | psf | US-MRI fusion |
| waxs | done | done | synthetic | done | identity | WAXS pattern |
| weather_radar | done | done | synthetic | done | psf | Radar beam |
| widefield_lowdose | done | done | synthetic | done | psf | Widefield low-dose |
| xfel_sfx | done | done | synthetic | done | identity | XFEL diffraction |
| xray_crystallography | done | done | synthetic | done | identity | X-ray diffraction |
| xray_ndt | done | done | synthetic | done | radon | X-ray NDT projection |
| xray_radiography | done | done | synthetic | done | radon | X-ray projection |
| xrf_imaging | done | done | synthetic | done | identity | XRF spectral |
| xrf_tomo | done | done | synthetic | done | radon | XRF tomography |

---

## Data Quality Audit Summary (2026-03-11)

| Check | Result |
|-------|--------|
| Total modalities audited | 169 |
| PASS (all checks OK) | 160 |
| WARN (non-standard keys, data valid) | 6 (insar, mammography, oct, spect_ct, spectral_ct, ultrasound) |
| FAIL (missing standard keys, but have domain-specific alternatives) | 3 (holography, pet_ct, pet_mr) |
| Pending (no dataset yet) | 2 (cassi, spc) |
| NaN/Inf detected | 0 |
| All-zero x_true detected | 0 |

**Notes on WARN/FAIL modalities** — all have valid data with domain-appropriate key names:
- `holography`: x_true_amplitude + x_true_phase (complex field split)
- `pet_ct`: x_ct + x_pet, y_ct + y_pet (dual modality)
- `pet_mr`: x_mr + x_pet, y_mr + y_pet (dual modality, complex k-space)
- `insar`: y_real + y_imag (complex wrapped interferogram)
- `mammography`: projection_measured (Beer-Lambert forward model)
- `oct`: bscan_measured (OCT B-scan)
- `spect_ct`: y_ct + y_spect (dual modality)
- `spectral_ct`: y_low + y_high (dual energy)
- `ultrasound`: bmode_measured (B-mode ultrasound)

## Data Source Summary

| Source Type | Count | Description |
|-------------|-------|-------------|
| **real** | 4 | ct (LoDoPaB-CT), mri (M4Raw), sd_cassi (KAIST TSA), cacti (real video) |
| **domain-sim** | 37 | Dedicated generators with physics-specific forward models |
| **synthetic** | 126 | Universal generator with domain-appropriate phantoms (incl. hyperspectral_remote, multispectral_sat, nerf) |
| **pending** | 2 | spc, cassi |
