# Use Case 1: Reconstruct — Full Modality Index

All 169 modalities available in `algorithm_base/`. Full spec files exist for the most common ones.

## Quick Index by Category

### Medical Imaging — Tomography

| Modality ID | Full Name | Best CPU PSNR | Best GPU PSNR | Spec File |
|-------------|-----------|---------------|---------------|-----------|
| `ct` | X-ray CT | 39.5 dB (PnP-NLM) | 43.5 dB (InDuDoNet) | [ct.md](ct.md) |
| `cbct` | Cone-Beam CT | 37 dB (TV-ADMM 3D) | 38+ dB | [cbct.md](cbct.md) |
| `pet` | Positron Emission Tomography | 30 dB (OSEM) | 35+ dB | [pet.md](pet.md) |
| `spect` | Single Photon Emission CT | 28 dB (OSEM) | — | [spect.md](spect.md) |
| `spect_ct` | SPECT/CT Fusion | — | — | _template |
| `pet_ct` | PET/CT Fusion | — | — | _template |
| `pet_mr` | PET/MR Fusion | — | — | _template |
| `spectral_ct` | Photon-Counting Spectral CT | — | — | _template |
| `angiography` | X-ray Angiography / DSA | — | — | _template |
| `mammography` | Mammography | — | — | _template |
| `fluoroscopy` | Fluoroscopy | — | — | _template |
| `digital_breast_tomo` | Digital Breast Tomosynthesis | — | — | _template |
| `xray_radiography` | X-ray Radiography | — | — | _template |
| `industrial_ct` | Industrial CT (NDT) | — | — | _template |
| `neutron_tomo` | Neutron Tomography | — | — | _template |
| `muon_tomo` | Muon Tomography | — | — | _template |
| `electron_tomography` | Electron Tomography | — | — | _template |
| `portal_imaging` | Megavoltage Portal Imaging | — | — | _template |
| `brachytherapy_img` | Brachytherapy Imaging | — | — | _template |
| `proton_therapy_img` | Proton Therapy Imaging | — | — | _template |
| `proton_radiography` | Proton Radiography | — | — | _template |

### Medical Imaging — MRI

| Modality ID | Full Name | Best CPU PSNR | Best GPU PSNR | Spec File |
|-------------|-----------|---------------|---------------|-----------|
| `mri` | MRI | 34.2 dB (ESPIRiT) | 37.8 dB (ReconFormer) | [mri.md](mri.md) |
| `diffusion_mri` | Diffusion MRI | — | — | _template |
| `fmri` | Functional MRI | — | — | _template |
| `asl_mri` | Arterial Spin Labeling | — | — | _template |
| `cest_mri` | CEST MRI | — | — | _template |
| `mr_elastography` | MR Elastography | — | — | _template |
| `mr_fingerprinting` | MR Fingerprinting | — | — | _template |
| `mra` | MR Angiography | — | — | _template |
| `mrs` | MR Spectroscopy | — | — | _template |
| `swi` | Susceptibility Weighted Imaging | — | — | _template |
| `us_mri` | Undersampled MRI | — | — | _template |

### Medical Imaging — Ultrasound

| Modality ID | Full Name | Best CPU PSNR | Spec File |
|-------------|-----------|---------------|-----------|
| `ultrasound` | Ultrasound (B-mode) | 32 dB (DAS) | [ultrasound.md](ultrasound.md) |
| `doppler_ultrasound` | Doppler Ultrasound | — | _template |
| `elastography` | Ultrasound Elastography | — | _template |
| `ivus` | Intravascular Ultrasound | — | _template |
| `ultrasonic_phased_array` | Phased Array NDT | — | _template |
| `ceus` | Contrast-Enhanced Ultrasound | — | _template |

### Medical Imaging — Optical

| Modality ID | Full Name | Best CPU PSNR | Best GPU PSNR | Spec File |
|-------------|-----------|---------------|---------------|-----------|
| `oct` | Optical Coherence Tomography | 35 dB (TV) | 40+ dB | [oct.md](oct.md) |
| `octa` | OCT Angiography | — | — | _template |
| `fundus` | Fundus Photography | — | — | _template |
| `endoscopy` | Endoscopy | — | — | _template |
| `photoacoustic` | Photoacoustic Imaging | — | — | _template |
| `dot` | Diffuse Optical Tomography | — | — | _template |
| `confocal_endomicroscopy` | Confocal Endomicroscopy | — | — | _template |

### Electron Microscopy

| Modality ID | Full Name | Best CPU PSNR | Best GPU PSNR | Spec File |
|-------------|-----------|---------------|---------------|-----------|
| `cryo_em` | Cryo-EM | — | — | _template |
| `tem` | TEM | — | — | _template |
| `stem` | STEM | — | — | _template |
| `sem` | SEM | — | — | _template |
| `cryo_et` | Cryo-ET | — | — | _template |
| `fib_sem` | FIB-SEM | — | — | _template |
| `eels` | EELS | — | — | _template |
| `edx_mapping` | EDX Mapping | — | — | _template |
| `ebsd` | EBSD | — | — | _template |
| `electron_diffraction` | Electron Diffraction | — | — | _template |
| `electron_holography` | Electron Holography | — | — | _template |
| `eht_imaging` | EHT / In-situ EM | — | — | _template |

### Optical Microscopy

| Modality ID | Full Name | Best CPU PSNR | Best GPU PSNR | Spec File |
|-------------|-----------|---------------|---------------|-----------|
| `widefield` | Widefield Fluorescence | 33 dB (RL) | 38+ dB | _template |
| `widefield_lowdose` | Low-Dose Widefield | — | — | _template |
| `confocal_3d` | Confocal 3D | — | — | _template |
| `confocal_livecell` | Live-Cell Confocal | — | — | _template |
| `two_photon` | Two-Photon Microscopy | — | — | _template |
| `three_photon` | Three-Photon Microscopy | — | — | _template |
| `lattice_lightsheet` | Lattice Light Sheet | — | — | _template |
| `lightsheet` | Light Sheet | — | — | _template |
| `tirf` | TIRF Microscopy | — | — | _template |
| `spinning_disk` | Spinning Disk Confocal | — | — | _template |
| `flim` | FLIM | — | — | _template |

### Super-Resolution Microscopy

| Modality ID | Full Name | Best CPU PSNR | Best GPU PSNR | Spec File |
|-------------|-----------|---------------|---------------|-----------|
| `sim` | Structured Illumination | 37 dB (wienerSIM) | 42 dB (DFCAN) | [sim.md](sim.md) |
| `sted` | STED | — | — | _template |
| `palm_storm` | PALM/STORM | — | — | _template |
| `minflux` | MINFLUX | — | — | _template |
| `dna_paint` | DNA-PAINT | — | — | _template |
| `expansion` | Expansion Microscopy | — | — | _template |

### Spectroscopy

| Modality ID | Full Name | Spec File |
|-------------|-----------|-----------|
| `raman_imaging` | Raman Imaging | _template |
| `cars` | CARS Microscopy | _template |
| `srs` | SRS Microscopy | _template |
| `ftir_imaging` | FTIR Imaging | _template |
| `maldi_msi` | MALDI Mass Spec Imaging | _template |
| `desi` | DESI MSI | _template |
| `saxs` | SAXS | _template |
| `waxs` | WAXS | _template |
| `xrf_imaging` | XRF Imaging | _template |
| `xrf_tomo` | XRF Tomography | _template |
| `xfel_sfx` | XFEL Serial Crystallography | _template |
| `xray_crystallography` | X-ray Crystallography | _template |
| `brillouin` | Brillouin Microscopy | _template |
| `cathodoluminescence` | Cathodoluminescence | _template |
| `sims` | SIMS | _template |
| `libs` | LIBS | _template |
| `shg` | SHG Microscopy | _template |

### Computational Imaging

| Modality ID | Full Name | Best CPU PSNR | Best GPU PSNR | Spec File |
|-------------|-----------|---------------|---------------|-----------|
| `cassi` | Coded-Aperture Spectral | 38 dB (GAP-TV) | 42 dB (MST-L) | [cassi.md](cassi.md) |
| `cacti` | Compressed Video | 35 dB (GAP-TV) | 40 dB (EfficientSCI) | [cacti.md](cacti.md) |
| `lensless` | Lensless Imaging | 32 dB (ADMM-TV) | 38 dB (PhysenNet) | [lensless.md](lensless.md) |
| `phase_retrieval` | Phase Retrieval | 36 dB (HIO+ER) | 40 dB | [phase_retrieval.md](phase_retrieval.md) |
| `spc` | Single Pixel Camera | — | — | _template |
| `ghost_imaging` | Ghost Imaging | — | — | _template |
| `ptychography` | Ptychography | — | — | _template |
| `holography` | Digital Holography | — | — | _template |
| `fpm` | Fourier Ptychographic Microscopy | — | — | _template |
| `odt` | Optical Diffraction Tomography | — | — | _template |
| `light_field` | Light Field Imaging | — | — | _template |
| `integral` | Integral Imaging | — | — | _template |
| `cup` | Compressed Ultrafast Photography | — | — | _template |
| `streak_camera` | Streak Camera | — | — | _template |
| `coded_exposure` | Coded Exposure | — | — | _template |
| `event_camera` | Event Camera | — | — | _template |
| `hdr_imaging` | HDR Imaging | — | — | _template |
| `structured_light` | Structured Light 3D | — | — | _template |
| `tof_camera` | Time-of-Flight Camera | — | — | _template |
| `nerf` | Neural Radiance Fields | — | — | _template |
| `gaussian_splatting` | Gaussian Splatting | — | — | _template |

### Remote Sensing

| Modality ID | Full Name | Spec File |
|-------------|-----------|-----------|
| `sar` | SAR Imaging | _template |
| `insar` | InSAR | _template |
| `polsar` | PolSAR | _template |
| `hyperspectral_remote` | Hyperspectral Remote Sensing | _template |
| `multispectral_sat` | Multispectral Satellite | _template |
| `ocean_color` | Ocean Color | _template |
| `passive_microwave` | Passive Microwave | _template |
| `weather_radar` | Weather Radar | _template |
| `lidar` | LiDAR | _template |
| `flash_lidar` | Flash LiDAR | _template |

### Geophysics / Seismic

| Modality ID | Full Name | Spec File |
|-------------|-----------|-----------|
| `seismic_tomo` | Seismic Tomography | _template |
| `fwi` | Full Waveform Inversion | _template |
| `gpr` | Ground Penetrating Radar | _template |
| `impedance_tomo` | Electrical Impedance Tomography | _template |
| `sonar` | Sonar Imaging | _template |
| `ocean_acoustic_tomo` | Ocean Acoustic Tomography | _template |

### Astronomy / Physics

| Modality ID | Full Name | Spec File |
|-------------|-----------|-----------|
| `radio_astronomy` | Radio Astronomy Imaging | _template |
| `radio_interferometry` | Radio Interferometry | _template |
| `eht_imaging` | Event Horizon Telescope | _template |
| `coronagraphy` | Coronagraphy | _template |
| `solar_imaging` | Solar Imaging | _template |
| `gravitational_wave` | Gravitational Wave | _template |
| `particle_calorimetry` | Particle Calorimetry | _template |
| `quantum_illumination` | Quantum Illumination | _template |
| `entangled_photon` | Entangled Photon Imaging | _template |
| `lucky_imaging` | Lucky Imaging | _template |

### NDT / Industrial

| Modality ID | Full Name | Spec File |
|-------------|-----------|-----------|
| `xray_ndt` | X-ray NDT | _template |
| `active_thermography` | Active Thermography | _template |
| `eddy_current` | Eddy Current Imaging | _template |
| `ultrasonic_phased_array` | Ultrasonic Phased Array | _template |
| `shearography` | Shearography | _template |

### SPM / Nanoscale

| Modality ID | Full Name | Spec File |
|-------------|-----------|-----------|
| `afm` | Atomic Force Microscopy | _template |
| `stm` | Scanning Tunneling Microscopy | _template |
| `mfm` | Magnetic Force Microscopy | _template |
| `nsom` | Near-Field Optical Microscopy | _template |

### Other

| Modality ID | Full Name | Spec File |
|-------------|-----------|-----------|
| `dexa` | Dual-Energy X-ray Absorptiometry | _template |
| `magnetic_particle` | Magnetic Particle Imaging | _template |
| `bioluminescence_tomo` | Bioluminescence Tomography | _template |
| `clem` | CLEM | _template |
| `adaptive_optics` | Adaptive Optics | _template |
| `dic` | DIC Microscopy | _template |
| `terahertz` | Terahertz Imaging | _template |
| `polarization` | Polarization Imaging | _template |
| `pump_probe` | Pump-Probe Spectroscopy | _template |
| `photometric_stereo` | Photometric Stereo | _template |
| `panorama` | Panoramic Imaging | _template |
| `machine_vision` | Machine Vision | _template |
| `atom_probe` | Atom Probe Tomography | _template |
| `dark_field` | Dark-Field Imaging | _template |
| `phase_contrast` | Phase Contrast Imaging | _template |
| `talbot_lau` | Talbot-Lau Interferometry | _template |
| `ct_fluorescence` | CT-Fluorescence Fusion | _template |
| `ism` | Image Scanning Microscopy | _template |
| `acoustic_microscopy` | Acoustic Microscopy | _template |
| `acoustic_emission` | Acoustic Emission | _template |
| `matrix` | Matrix Imaging | _template |

---

## How to Use This Index

**With keyword_match.py:**
```bash
python spec/keyword_match.py "CT reconstruction"
python spec/keyword_match.py "hyperspectral imaging" --run
```

**Directly in Python:**
```python
import sys
sys.path.insert(0, 'path/to/Physics_World_Model/pwm/public')
sys.path.insert(0, 'path/to/Physics_World_Model/pwm/public/packages/pwm_core')
from algorithm_base._registry import list_modalities, list_solvers, run_solver

# List all 169 modalities
print(list_modalities())

# List solvers for any modality
print(list_solvers('ct'))

# Run any solver
import numpy as np
y = np.random.rand(180, 256).astype(np.float32)
x_hat = run_solver('ct', 'traditional_cpu', y)
```
