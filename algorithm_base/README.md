# PWM Algorithm Base

168 modalities, 541 solvers — organized for server, paper, and CLI usage.

## Quick Start

```python
# 1. Import by modality name
from algorithm_base.cassi import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# 2. Use specific solver function
from algorithm_base.cacti import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)

# 3. Via central registry
from algorithm_base import get_solver, list_solvers, list_modalities

# List all 168 modalities
for mod in list_modalities():
    print(mod)

# List solvers for a modality
for key, info in list_solvers("mri"):
    print(f"{key}: {info['name']} (GPU={info['gpu']})")

# Get and run a solver
solver_fn = get_solver("ct", "traditional_cpu")
x_hat = solver_fn(sinogram, radon_op, {"iterations": 50})

# Or run directly
from algorithm_base import run_solver
x_hat = run_solver("mri", "best_quality", kspace, mri_op)
```

## Three Use Cases

### 1. Main Server (no GPU)
```python
from algorithm_base import run_solver, list_solvers

# CPU solvers run directly
x_hat = run_solver("cassi", "traditional_cpu", y, op)

# For GPU solvers, use SolverDispatcher with Modal
from pwm_core.recon.solver_dispatch import SolverDispatcher
dispatcher = SolverDispatcher(use_modal=True)
x_hat, meta = dispatcher.run("cacti", "hisvit13", y, op)
```

### 2. Paper Benchmarks
```python
from algorithm_base import list_modalities, list_solvers, run_solver

for modality in list_modalities():
    for key, info in list_solvers(modality):
        x_hat = run_solver(modality, key, y_dict[modality], op_dict[modality])
        # compute PSNR, SSIM for paper table
```

### 3. PWM CLI
```bash
pwm evaluate --method traditional_cpu --modality cassi --track correct
```

## Adding New Algorithms

1. Create solver function in `packages/pwm_core/pwm_core/recon/my_solver.py`:
```python
def run_my_solver(y, operator, cfg):
    # your reconstruction code
    return x_hat
```

2. Register in modality YAML (`benchmarks/configs/{modality}.yaml`):
```yaml
solvers:
  my_solver:
    name: "My Solver"
    module: "pwm_core.recon.my_solver"
    function: "run_my_solver"
    params:
      iterations: 100
    gpu: false
    reference: "Author et al., 2026"
```

3. Rebuild algorithm base:
```bash
python scripts/build_algorithm_base.py
```

## Directory Structure

```
algorithm_base/
  __init__.py          # from ._registry import get_solver, ...
  _registry.py         # Central dispatch: get_solver(), list_solvers(), ...
  README.md            # This file
  cassi/
    __init__.py        # from .solvers import run_solver, run_traditional_cpu, ...
    solvers.py         # SOLVERS dict + run_solver() + per-key functions
    README.md          # Modality info + algorithm leaderboard
  cacti/
    ...
  mri/
    ...
  (168 modality folders)
```

## Modality Index

| Modality | Display Name | Category | Solvers |
|----------|-------------|----------|---------|
| [`acoustic_emission`](./acoustic_emission/) | Acoustic Emission Testing (AE) | Broader Experimental Science | 3 |
| [`acoustic_microscopy`](./acoustic_microscopy/) | Scanning Acoustic Microscopy (SAM) | Industrial Inspection | 3 |
| [`active_thermography`](./active_thermography/) | Active Thermography (IR) | Industrial Inspection | 3 |
| [`adaptive_optics`](./adaptive_optics/) | Adaptive Optics (AO) Imaging | Broader Experimental Science | 3 |
| [`afm`](./afm/) | Atomic Force Microscopy (AFM) | Scanning Probe Microscopy | 3 |
| [`angiography`](./angiography/) | X-ray Angiography | Medical Imaging | 3 |
| [`asl_mri`](./asl_mri/) | Arterial Spin Labeling (ASL) MRI | Medical Imaging | 3 |
| [`atom_probe`](./atom_probe/) | Atom Probe Tomography (APT) | Scientific Instrumentation | 3 |
| [`bioluminescence_tomo`](./bioluminescence_tomo/) | Bioluminescence Tomography (BLT) | Broader Experimental Science | 3 |
| [`brachytherapy_img`](./brachytherapy_img/) | Brachytherapy Imaging | Medical Imaging | 3 |
| [`brillouin`](./brillouin/) | Brillouin Microscopy | Spectroscopy & Spectral Imaging | 3 |
| [`cacti`](./cacti/) | Coded Aperture Compressive Temporal Imaging (CACTI) | Compressive Imaging | 7 |
| [`cars`](./cars/) | Coherent Anti-Stokes Raman (CARS) Microscopy | Spectroscopy & Spectral Imaging | 3 |
| [`cassi`](./cassi/) | Coded Aperture Snapshot Spectral Imaging (CASSI) | Compressive Imaging | 7 |
| [`cathodoluminescence`](./cathodoluminescence/) | Cathodoluminescence (CL) Imaging | Scientific Instrumentation | 3 |
| [`cbct`](./cbct/) | Cone-Beam Computed Tomography (CBCT) | Medical Imaging | 3 |
| [`cest_mri`](./cest_mri/) | CEST MRI | Medical Imaging | 3 |
| [`ceus`](./ceus/) | Contrast-Enhanced Ultrasound (CEUS) | Medical Imaging | 3 |
| [`clem`](./clem/) | Correlative Light-Electron Microscopy (CLEM) | Multi-Modal Fusion | 3 |
| [`coded_exposure`](./coded_exposure/) | Coded Exposure / Flutter Shutter | Computational Photography | 3 |
| [`confocal_3d`](./confocal_3d/) | Confocal 3D Z-Stack | Microscopy | 4 |
| [`confocal_endomicroscopy`](./confocal_endomicroscopy/) | Confocal Laser Endomicroscopy (CLE) | Medical Imaging | 3 |
| [`confocal_livecell`](./confocal_livecell/) | Confocal Live-Cell Microscopy | Microscopy | 4 |
| [`coronagraphy`](./coronagraphy/) | Stellar Coronagraphy | Astronomy & Space Imaging | 3 |
| [`cryo_em`](./cryo_em/) | Cryo-EM Single Particle Analysis | Scientific Instrumentation | 3 |
| [`cryo_et`](./cryo_et/) | Cryo-Electron Tomography (Cryo-ET) | Electron Microscopy | 3 |
| [`ct`](./ct/) | X-ray Computed Tomography (CT) | Medical Imaging | 4 |
| [`ct_fluorescence`](./ct_fluorescence/) | CT + Fluorescence (FLIT) | Multi-Modal Fusion | 3 |
| [`cup`](./cup/) | Compressed Ultrafast Photography (CUP) | Ultrafast Imaging | 3 |
| [`dark_field`](./dark_field/) | Dark-Field Microscopy | Microscopy | 3 |
| [`desi`](./desi/) | DESI Mass Spectrometry Imaging | Spectroscopy & Spectral Imaging | 3 |
| [`dexa`](./dexa/) | Dual-Energy X-ray Absorptiometry (DEXA) | Medical Imaging | 3 |
| [`dic`](./dic/) | Differential Interference Contrast (DIC) | Microscopy | 3 |
| [`diffusion_mri`](./diffusion_mri/) | Diffusion MRI (DTI) | Medical Imaging | 3 |
| [`digital_breast_tomo`](./digital_breast_tomo/) | Digital Breast Tomosynthesis (DBT) | Medical Imaging | 3 |
| [`dna_paint`](./dna_paint/) | DNA-PAINT Super-Resolution | Microscopy | 3 |
| [`doppler_ultrasound`](./doppler_ultrasound/) | Doppler Ultrasound | Medical Imaging | 3 |
| [`dot`](./dot/) | Diffuse Optical Tomography (DOT) | Medical Imaging | 3 |
| [`ebsd`](./ebsd/) | Electron Backscatter Diffraction (EBSD) | Electron Microscopy | 3 |
| [`eddy_current`](./eddy_current/) | Eddy Current Imaging | Industrial Inspection | 3 |
| [`edx_mapping`](./edx_mapping/) | STEM-EDX Elemental Mapping | Electron Microscopy | 3 |
| [`eels`](./eels/) | Electron Energy Loss Spectroscopy (EELS) | Electron Microscopy | 4 |
| [`eht_imaging`](./eht_imaging/) | Event Horizon Telescope (EHT) Imaging | Astronomy & Space Imaging | 3 |
| [`elastography`](./elastography/) | Shear-Wave Elastography | Medical Imaging | 3 |
| [`electron_diffraction`](./electron_diffraction/) | 4D-STEM Electron Diffraction | Electron Microscopy | 3 |
| [`electron_holography`](./electron_holography/) | Electron Holography | Electron Microscopy | 3 |
| [`electron_tomography`](./electron_tomography/) | Electron Tomography | Electron Microscopy | 3 |
| [`endoscopy`](./endoscopy/) | Fiber Bundle Endoscopy | Medical Imaging | 3 |
| [`entangled_photon`](./entangled_photon/) | Entangled Photon Microscopy | Quantum Imaging | 3 |
| [`event_camera`](./event_camera/) | Event Camera / Dynamic Vision Sensor (DVS) | Computational Photography | 3 |
| [`expansion`](./expansion/) | Expansion Microscopy (ExM) | Microscopy | 3 |
| [`fib_sem`](./fib_sem/) | Focused Ion Beam SEM (FIB-SEM) | Electron Microscopy | 3 |
| [`flash_lidar`](./flash_lidar/) | Flash LiDAR | Depth Imaging | 3 |
| [`flim`](./flim/) | Fluorescence Lifetime Imaging (FLIM) | Microscopy | 4 |
| [`fluoroscopy`](./fluoroscopy/) | Fluoroscopy | Medical Imaging | 3 |
| [`fmri`](./fmri/) | Functional MRI (BOLD fMRI) | Medical Imaging | 3 |
| [`fpm`](./fpm/) | Fourier Ptychographic Microscopy (FPM) | Microscopy | 4 |
| [`ftir_imaging`](./ftir_imaging/) | FTIR Spectroscopic Imaging | Spectroscopy & Spectral Imaging | 3 |
| [`fundus`](./fundus/) | Fundus Camera | Medical Imaging | 3 |
| [`fwi`](./fwi/) | Full-Waveform Inversion (FWI) | Broader Experimental Science | 3 |
| [`gaussian_splatting`](./gaussian_splatting/) | 3D Gaussian Splatting (3DGS) | Neural Rendering | 4 |
| [`ghost_imaging`](./ghost_imaging/) | Ghost Imaging | Quantum Imaging | 3 |
| [`gpr`](./gpr/) | Ground-Penetrating Radar (GPR) | Remote Sensing | 3 |
| [`gravitational_wave`](./gravitational_wave/) | Gravitational Wave Detection | Broader Experimental Science | 3 |
| [`hdr_imaging`](./hdr_imaging/) | High Dynamic Range (HDR) Imaging | Computational Photography | 3 |
| [`holography`](./holography/) | Digital Holographic Microscopy | Coherent Imaging | 4 |
| [`hyperspectral_remote`](./hyperspectral_remote/) | Hyperspectral Remote Sensing | Remote Sensing | 3 |
| [`impedance_tomo`](./impedance_tomo/) | Electrical Impedance Tomography (EIT) | Broader Experimental Science | 3 |
| [`industrial_ct`](./industrial_ct/) | Industrial X-ray CT | Industrial Inspection | 3 |
| [`insar`](./insar/) | Interferometric SAR (InSAR) | Remote Sensing | 3 |
| [`integral`](./integral/) | Integral Photography | Computational Optics | 4 |
| [`ism`](./ism/) | Image Scanning Microscopy (ISM) | Microscopy | 3 |
| [`ivus`](./ivus/) | Intravascular Ultrasound (IVUS) | Medical Imaging | 3 |
| [`lattice_lightsheet`](./lattice_lightsheet/) | Lattice Light-Sheet Microscopy | Microscopy | 3 |
| [`lensless`](./lensless/) | Lensless (Diffuser Camera) Imaging | Computational Photography | 4 |
| [`libs`](./libs/) | Laser-Induced Breakdown Spectroscopy (LIBS) Imaging | Spectroscopy & Spectral Imaging | 3 |
| [`lidar`](./lidar/) | LiDAR Scanner | Depth Imaging | 3 |
| [`light_field`](./light_field/) | Light Field Imaging | Computational Optics | 4 |
| [`lightsheet`](./lightsheet/) | Light-Sheet Fluorescence Microscopy (LSFM) | Microscopy | 4 |
| [`lucky_imaging`](./lucky_imaging/) | Lucky Imaging | Astronomy & Space Imaging | 3 |
| [`machine_vision`](./machine_vision/) | Machine Vision / AOI | Industrial Inspection | 3 |
| [`magnetic_particle`](./magnetic_particle/) | Magnetic Particle Imaging (MPI) | Broader Experimental Science | 3 |
| [`maldi_msi`](./maldi_msi/) | MALDI Mass Spectrometry Imaging | Scientific Instrumentation | 3 |
| [`mammography`](./mammography/) | Mammography | Medical Imaging | 3 |
| [`matrix`](./matrix/) | Generic Matrix Sensing | Compressive Imaging | 4 |
| [`mfm`](./mfm/) | Magnetic Force Microscopy (MFM) | Scanning Probe Microscopy | 3 |
| [`minflux`](./minflux/) | MINFLUX Nanoscopy | Microscopy | 3 |
| [`mr_elastography`](./mr_elastography/) | MR Elastography (MRE) | Medical Imaging | 3 |
| [`mr_fingerprinting`](./mr_fingerprinting/) | MR Fingerprinting (MRF) | Medical Imaging | 3 |
| [`mra`](./mra/) | MR Angiography (MRA) | Medical Imaging | 3 |
| [`mri`](./mri/) | Magnetic Resonance Imaging (MRI) | Medical Imaging | 5 |
| [`mrs`](./mrs/) | MR Spectroscopy (MRS) | Medical Imaging | 3 |
| [`multispectral_sat`](./multispectral_sat/) | Multispectral Satellite Imaging | Remote Sensing | 3 |
| [`muon_tomo`](./muon_tomo/) | Muon Tomography | Scientific Instrumentation | 3 |
| [`nerf`](./nerf/) | Neural Radiance Fields (NeRF) | Neural Rendering | 6 |
| [`neutron_diffraction`](./neutron_diffraction/) | Neutron Diffraction | Scientific Instrumentation | 3 |
| [`neutron_tomo`](./neutron_tomo/) | Neutron Radiography / Tomography | Scientific Instrumentation | 3 |
| [`nirs_brain`](./nirs_brain/) | Functional Near-Infrared Spectroscopy (fNIRS) | Medical Imaging | 3 |
| [`nsom`](./nsom/) | Near-field Scanning Optical Microscopy (NSOM) | Scanning Probe Microscopy | 3 |
| [`ocean_acoustic_tomo`](./ocean_acoustic_tomo/) | Ocean Acoustic Tomography | Broader Experimental Science | 3 |
| [`ocean_color`](./ocean_color/) | Ocean Color Remote Sensing | Remote Sensing | 3 |
| [`oct`](./oct/) | Optical Coherence Tomography (OCT) | Medical Imaging | 4 |
| [`octa`](./octa/) | OCT Angiography (OCTA) | Medical Imaging | 3 |
| [`odt`](./odt/) | Optical Diffraction Tomography (ODT) | Coherent Imaging | 3 |
| [`palm_storm`](./palm_storm/) | PALM/STORM Single-Molecule Localization | Microscopy | 3 |
| [`panorama`](./panorama/) | Panorama Multi-Focus Fusion | Computational Photography | 4 |
| [`particle_calorimetry`](./particle_calorimetry/) | Particle Calorimetry | Broader Experimental Science | 3 |
| [`passive_microwave`](./passive_microwave/) | Passive Microwave Radiometry | Remote Sensing | 3 |
| [`pet`](./pet/) | Positron Emission Tomography (PET) | Medical Imaging | 3 |
| [`pet_ct`](./pet_ct/) | PET/CT Fusion | Multi-Modal Fusion | 3 |
| [`pet_mr`](./pet_mr/) | PET/MR Fusion | Multi-Modal Fusion | 3 |
| [`phase_contrast`](./phase_contrast/) | Phase Contrast Microscopy | Microscopy | 3 |
| [`phase_retrieval`](./phase_retrieval/) | Coherent Diffractive Imaging / Phase Retrieval | Coherent Imaging | 4 |
| [`photoacoustic`](./photoacoustic/) | Photoacoustic Imaging | Medical Imaging | 4 |
| [`photometric_stereo`](./photometric_stereo/) | Photometric Stereo | Depth Imaging | 3 |
| [`polarization`](./polarization/) | Polarization Microscopy | Microscopy | 3 |
| [`polsar`](./polsar/) | Polarimetric SAR (PolSAR) | Remote Sensing | 3 |
| [`portal_imaging`](./portal_imaging/) | Portal Imaging (EPID) | Medical Imaging | 3 |
| [`proton_radiography`](./proton_radiography/) | Proton Radiography | Scientific Instrumentation | 3 |
| [`proton_therapy_img`](./proton_therapy_img/) | Proton Therapy Imaging | Medical Imaging | 3 |
| [`ptychography`](./ptychography/) | Ptychographic Imaging | Coherent Imaging | 4 |
| [`pump_probe`](./pump_probe/) | Pump-Probe Microscopy | Ultrafast Imaging | 3 |
| [`quantum_illumination`](./quantum_illumination/) | Quantum Illumination | Quantum Imaging | 3 |
| [`radio_astronomy`](./radio_astronomy/) | Radio Aperture Synthesis | Broader Experimental Science | 3 |
| [`radio_interferometry`](./radio_interferometry/) | Radio Interferometry (VLBI) | Remote Sensing | 3 |
| [`raman_imaging`](./raman_imaging/) | Raman Imaging / Microscopy | Spectroscopy & Spectral Imaging | 3 |
| [`sar`](./sar/) | Synthetic Aperture Radar (SAR) | Remote Sensing | 3 |
| [`saxs`](./saxs/) | Small-Angle X-ray Scattering (SAXS) | Scientific Instrumentation | 3 |
| [`seismic_tomo`](./seismic_tomo/) | Seismic Tomography | Broader Experimental Science | 3 |
| [`sem`](./sem/) | Scanning Electron Microscopy (SEM) | Electron Microscopy | 3 |
| [`shearography`](./shearography/) | Shearography | Industrial Inspection | 3 |
| [`shg`](./shg/) | Second Harmonic Generation (SHG) Microscopy | Microscopy | 3 |
| [`sim`](./sim/) | Structured Illumination Microscopy (SIM) | Microscopy | 4 |
| [`sims`](./sims/) | Secondary Ion Mass Spectrometry (SIMS) Imaging | Spectroscopy & Spectral Imaging | 3 |
| [`solar_imaging`](./solar_imaging/) | Solar EUV/X-ray Imaging | Astronomy & Space Imaging | 3 |
| [`sonar`](./sonar/) | Sonar Imaging | Remote Sensing | 3 |
| [`spc`](./spc/) | Single-Pixel Camera (SPC) | Compressive Imaging | 6 |
| [`spect`](./spect/) | Single Photon Emission CT (SPECT) | Medical Imaging | 3 |
| [`spect_ct`](./spect_ct/) | SPECT/CT Fusion | Multi-Modal Fusion | 3 |
| [`spectral_ct`](./spectral_ct/) | Photon-Counting Spectral CT | Medical Imaging | 3 |
| [`spinning_disk`](./spinning_disk/) | Spinning Disk Confocal Microscopy | Microscopy | 3 |
| [`srs`](./srs/) | Stimulated Raman Scattering (SRS) Microscopy | Spectroscopy & Spectral Imaging | 3 |
| [`sted`](./sted/) | STED Microscopy | Microscopy | 3 |
| [`stem`](./stem/) | Scanning Transmission Electron Microscopy (STEM) | Electron Microscopy | 3 |
| [`stm`](./stm/) | Scanning Tunneling Microscopy (STM) | Scanning Probe Microscopy | 3 |
| [`streak_camera`](./streak_camera/) | Streak Camera Imaging | Ultrafast Imaging | 3 |
| [`structured_light`](./structured_light/) | Structured-Light Depth Camera | Depth Imaging | 3 |
| [`swi`](./swi/) | Susceptibility-Weighted Imaging (SWI) | Medical Imaging | 3 |
| [`talbot_lau`](./talbot_lau/) | Talbot-Lau X-ray Grating Interferometry | Coherent Imaging | 3 |
| [`tem`](./tem/) | Transmission Electron Microscopy (TEM) | Electron Microscopy | 3 |
| [`terahertz`](./terahertz/) | Terahertz Imaging (THz) | Industrial Inspection | 3 |
| [`three_photon`](./three_photon/) | Three-Photon Microscopy | Microscopy | 3 |
| [`tirf`](./tirf/) | TIRF Microscopy | Microscopy | 3 |
| [`tof_camera`](./tof_camera/) | Time-of-Flight Depth Camera | Depth Imaging | 3 |
| [`two_photon`](./two_photon/) | Two-Photon / Multiphoton Microscopy | Microscopy | 3 |
| [`ultrasonic_phased_array`](./ultrasonic_phased_array/) | Ultrasonic Phased Array (TFM/FMC) | Industrial Inspection | 3 |
| [`ultrasound`](./ultrasound/) | Ultrasound B-mode Imaging | Medical Imaging | 3 |
| [`us_mri`](./us_mri/) | US/MRI Fusion | Multi-Modal Fusion | 3 |
| [`waxs`](./waxs/) | Wide-Angle X-ray Scattering (WAXS) | Scientific Instrumentation | 3 |
| [`weather_radar`](./weather_radar/) | Weather / Doppler Radar | Remote Sensing | 3 |
| [`widefield`](./widefield/) | Widefield Fluorescence Microscopy | Microscopy | 4 |
| [`widefield_lowdose`](./widefield_lowdose/) | Low-Dose Widefield Microscopy | Microscopy | 4 |
| [`xfel_sfx`](./xfel_sfx/) | XFEL Serial Femtosecond Crystallography (SFX) | Ultrafast Imaging | 3 |
| [`xray_crystallography`](./xray_crystallography/) | X-ray Crystallography | Scientific Instrumentation | 3 |
| [`xray_ndt`](./xray_ndt/) | X-ray NDT (Radiography) | Industrial Inspection | 3 |
| [`xray_radiography`](./xray_radiography/) | X-ray Radiography | Medical Imaging | 3 |
| [`xrf_imaging`](./xrf_imaging/) | X-ray Fluorescence (XRF) Imaging | Industrial Inspection | 3 |
| [`xrf_tomo`](./xrf_tomo/) | X-ray Fluorescence Tomography | Scientific Instrumentation | 3 |
