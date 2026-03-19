# Modality Templates Index

Comprehensive 7-step templates for all 168 imaging modalities in the PWM5 Benchmark.
Each template covers: (1) Verify Standard Dataset, (2) List All Algorithms, (3) Update Solvers, (4) Verify Each Algorithm, (5) Upload Checkpoints to GCS, (6) Upload Standard Dataset to GCS, (7) Push to GitHub.

---

## Implementation Tracking — 12 Flagship Paper Modalities

Each algorithm must be implemented at least **5 times** (5 independent verification runs on the standard dataset).
When all 5 runs are complete, the algorithm status is marked **done**.

Progress: **0 / 131 algorithms done** | Last updated: 2026-03-19

---

### 1. CASSI — Coded Aperture Snapshot Spectral Imaging (`cassi`)

| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|---|-----------|-------|-------|-------|-------|-------|--------|
| 1 | TwIST | — | — | — | — | — | pending |
| 2 | GAP-TV | — | — | — | — | — | pending |
| 3 | PnP-HSICNN | — | — | — | — | — | pending |
| 4 | ADMM-Net | — | — | — | — | — | pending |
| 5 | GAP-Net | — | — | — | — | — | pending |
| 6 | Lambda-Net | — | — | — | — | — | pending |
| 7 | TSA-Net | — | — | — | — | — | pending |
| 8 | DGSMP | — | — | — | — | — | pending |
| 9 | BIRNAT | — | — | — | — | — | pending |
| 10 | BiSRNet | — | — | — | — | — | pending |
| 11 | HDNet | — | — | — | — | — | pending |
| 12 | MST-L | — | — | — | — | — | pending |
| 13 | MST++ | — | — | — | — | — | pending |
| 14 | CST-L-Plus | — | — | — | — | — | pending |
| 15 | DAUHST-9stg | — | — | — | — | — | pending |
| 16 | RDLUF-MixS2-9stg | — | — | — | — | — | pending |
| 17 | SSR-L | — | — | — | — | — | pending |
| 18 | PADUT-3stg | — | — | — | — | — | pending |
| 19 | MiJUN-5stg | — | — | — | — | — | pending |

---

### 2. CACTI — Coded Aperture Compressive Temporal Imaging (`cacti`)

| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|---|-----------|-------|-------|-------|-------|-------|--------|
| 1 | GAP-TV | — | — | — | — | — | pending |
| 2 | PnP-FFDNet | — | — | — | — | — | pending |
| 3 | EfficientSCI | — | — | — | — | — | pending |
| 4 | EfficientSCI-T | — | — | — | — | — | pending |
| 5 | ELP-Unfolding | — | — | — | — | — | pending |
| 6 | HiSViT-9 | — | — | — | — | — | pending |
| 7 | HiSViT-13 | — | — | — | — | — | pending |

---

### 3. SPC — Single-Pixel Camera (`spc`)

| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|---|-----------|-------|-------|-------|-------|-------|--------|
| 1 | TVAL3 | — | — | — | — | — | pending |
| 2 | ADMM-L1 | — | — | — | — | — | pending |
| 3 | FISTA-L1 | — | — | — | — | — | pending |
| 4 | ISTA-Net+ | — | — | — | — | — | pending |
| 5 | HATNet | — | — | — | — | — | pending |

---

### 4. Lensless Imaging (`lensless`)

| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|---|-----------|-------|-------|-------|-------|-------|--------|
| 1 | ADMM-TV | — | — | — | — | — | pending |
| 2 | FlatNet | — | — | — | — | — | pending |
| 3 | FlatNet-Lite | — | — | — | — | — | pending |

---

### 5. Digital Holographic Microscopy / Compressive Holography (`holography`)

| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|---|-----------|-------|-------|-------|-------|-------|--------|
| 1 | Angular Spectrum | — | — | — | — | — | pending |
| 2 | PhaseNet | — | — | — | — | — | pending |

---

### 6. Ptychographic Imaging / Electron Ptychography (`ptychography`)

| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|---|-----------|-------|-------|-------|-------|-------|--------|
| 1 | ePIE | — | — | — | — | — | pending |
| 2 | PtychoNN | — | — | — | — | — | pending |
| 3 | PtychoNN 2.0 | — | — | — | — | — | pending |

---

### 7. CT — X-ray Computed Tomography (`ct`)

| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|---|-----------|-------|-------|-------|-------|-------|--------|
| 1 | FBP (Ram-Lak) | — | — | — | — | — | pending |
| 2 | FBP (Shepp-Logan) | — | — | — | — | — | pending |
| 3 | FBP (Cosine) | — | — | — | — | — | pending |
| 4 | FBP (Hamming) | — | — | — | — | — | pending |
| 5 | FBP (Hann) | — | — | — | — | — | pending |
| 6 | Landweber | — | — | — | — | — | pending |
| 7 | ART | — | — | — | — | — | pending |
| 8 | SIRT | — | — | — | — | — | pending |
| 9 | CGLS | — | — | — | — | — | pending |
| 10 | MLEM | — | — | — | — | — | pending |
| 11 | SART | — | — | — | — | — | pending |
| 12 | OSEM | — | — | — | — | — | pending |
| 13 | Tikhonov | — | — | — | — | — | pending |
| 14 | TV-ADMM | — | — | — | — | — | pending |
| 15 | Chambolle-Pock | — | — | — | — | — | pending |
| 16 | PnP-ADMM (NLM) | — | — | — | — | — | pending |
| 17 | PnP-HQS (NLM) | — | — | — | — | — | pending |
| 18 | PnP-FISTA (NLM) | — | — | — | — | — | pending |
| 19 | PnP-ADMM (BM3D) | — | — | — | — | — | pending |
| 20 | FBP + NLM | — | — | — | — | — | pending |
| 21 | FBP + BM3D | — | — | — | — | — | pending |
| 22 | FBP + Bilateral | — | — | — | — | — | pending |
| 23 | FBP + Wavelet | — | — | — | — | — | pending |
| 24 | FBP + TV | — | — | — | — | — | pending |
| 25 | RED-CNN | — | — | — | — | — | pending |
| 26 | FBPConvNet | — | — | — | — | — | pending |
| 27 | WGAN-VGG | — | — | — | — | — | pending |
| 28 | LEARN | — | — | — | — | — | pending |
| 29 | Learned Primal-Dual | — | — | — | — | — | pending |
| 30 | iRadonMAP | — | — | — | — | — | pending |
| 31 | FBP + U-Net | — | — | — | — | — | pending |
| 32 | DuDoNet | — | — | — | — | — | pending |
| 33 | InDuDoNet | — | — | — | — | — | pending |
| 34 | DuDoTrans | — | — | — | — | — | pending |
| 35 | CTformer | — | — | — | — | — | pending |
| 36 | Score-CT | — | — | — | — | — | pending |
| 37 | DPS | — | — | — | — | — | pending |
| 38 | DiffusionMBIR | — | — | — | — | — | pending |
| 39 | DOLCE | — | — | — | — | — | pending |
| 40 | CT-FM | — | — | — | — | — | pending |

---

### 8. CBCT — Cone-Beam CT (`cbct`)

| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|---|-----------|-------|-------|-------|-------|-------|--------|
| 1 | FDK / FBP | — | — | — | — | — | pending |
| 2 | FDK-DL | — | — | — | — | — | pending |
| 3 | CBCT-UNet | — | — | — | — | — | pending |

---

### 9. Ultrasound B-mode (`ultrasound`)

| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|---|-----------|-------|-------|-------|-------|-------|--------|
| 1 | Richardson-Lucy (ultrasound) | — | — | — | — | — | pending |
| 2 | US-UNet (DeepUS) | — | — | — | — | — | pending |
| 3 | US-CNN | — | — | — | — | — | pending |

---

### 10. Cryo-EM — Single-Particle Cryo-Electron Microscopy (`cryo_em`)

| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|---|-----------|-------|-------|-------|-------|-------|--------|
| 1 | Adjoint | — | — | — | — | — | pending |
| 2 | PnP-ADMM | — | — | — | — | — | pending |
| 3 | CryoDRGN | — | — | — | — | — | pending |

---

### 11. MRI — Magnetic Resonance Imaging (`mri`)

| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|---|-----------|-------|-------|-------|-------|-------|--------|
| 1 | Zero-Filled IFFT | — | — | — | — | — | pending |
| 2 | CS-MRI (Wavelet) | — | — | — | — | — | pending |
| 3 | CS-MRI (TV) | — | — | — | — | — | pending |
| 4 | SENSE | — | — | — | — | — | pending |
| 5 | ESPIRiT | — | — | — | — | — | pending |
| 6 | POCS | — | — | — | — | — | pending |
| 7 | ADMM | — | — | — | — | — | pending |
| 8 | Conjugate Gradient | — | — | — | — | — | pending |
| 9 | Truncated IFFT | — | — | — | — | — | pending |
| 10 | Gradient Descent | — | — | — | — | — | pending |
| 11 | Split Bregman | — | — | — | — | — | pending |
| 12 | GRAPPA-like | — | — | — | — | — | pending |
| 13 | ISTA | — | — | — | — | — | pending |
| 14 | FISTA | — | — | — | — | — | pending |
| 15 | Landweber Iteration | — | — | — | — | — | pending |
| 16 | Tikhonov Regularization | — | — | — | — | — | pending |
| 17 | Homodyne Detection | — | — | — | — | — | pending |
| 18 | PnP-ADMM | — | — | — | — | — | pending |
| 19 | Low-Rank (LORAKS) | — | — | — | — | — | pending |
| 20 | Nuclear Norm (SVT/SAKE) | — | — | — | — | — | pending |
| 21 | Proximal Gradient Descent | — | — | — | — | — | pending |
| 22 | BM3D-MRI | — | — | — | — | — | pending |
| 23 | SPIRiT-like | — | — | — | — | — | pending |
| 24 | RED (Regularization by Denoising) | — | — | — | — | — | pending |
| 25 | Dictionary Learning MRI | — | — | — | — | — | pending |
| 26 | ALOHA (Hankel Low-Rank) | — | — | — | — | — | pending |
| 27 | MoDL | — | — | — | — | — | pending |
| 28 | MoDL (5 unrolls) | — | — | — | — | — | pending |
| 29 | E2E-VarNet | — | — | — | — | — | pending |
| 30 | U-Net (fastMRI) | — | — | — | — | — | pending |
| 31 | DC-CNN | — | — | — | — | — | pending |
| 32 | Deep ADMM-Net | — | — | — | — | — | pending |
| 33 | ISTA-Net+ | — | — | — | — | — | pending |

---

### 12. Widefield Fluorescence Microscopy (`widefield`)

| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Status |
|---|-----------|-------|-------|-------|-------|-------|--------|
| 1 | Richardson-Lucy | — | — | — | — | — | pending |
| 2 | CARE | — | — | — | — | — | pending |

---

## Flagship Summary

| # | Modality | Algorithms | Done | Progress |
|---|----------|-----------|------|----------|
| 1 | CASSI | 19 | 0 | 0% |
| 2 | CACTI | 7 | 0 | 0% |
| 3 | SPC | 5 | 0 | 0% |
| 4 | Lensless | 3 | 0 | 0% |
| 5 | Holography | 2 | 0 | 0% |
| 6 | Ptychography | 3 | 0 | 0% |
| 7 | CT | 40 | 0 | 0% |
| 8 | CBCT | 3 | 0 | 0% |
| 9 | Ultrasound | 3 | 0 | 0% |
| 10 | Cryo-EM | 3 | 0 | 0% |
| 11 | MRI | 33 | 0 | 0% |
| 12 | Widefield | 2 | 0 | 0% |
| | **Total** | **123** | **0** | **0%** |

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

## Summary

| Location | Files | Modalities | Steps |
|----------|-------|------------|-------|
| `templates/` (this folder) | 5 | 55 | 385 |
| `_templates_part1–8.md` (parent) | 8 | 113 | 791 |
| **Total** | **13** | **168** | **1176** |

All 168 modalities now have complete 7-step templates.
