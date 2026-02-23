# Categories 5-19: Remaining Modalities — Full Case Catalog

> 127 modalities across 15 categories

---

## Category 5: Computational Photography (5 modalities)

### 5.1 Lensless (Diffuser Camera) Imaging

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **DiffuserCam** | Src(scene) -> C(random_diffuser_PSF) -> D(CMOS) | Polycarbonate diffuser + bare sensor |
| 2 | **PhlatCam** | Src(scene) -> C(designed_spiral_phase_mask) -> D(CMOS) | Optimized phase mask + sensor |
| 3 | **FlatCam/FlatScope** | Src(scene) -> M(binary_amplitude_mask) -> D(CMOS) | Random binary mask at 0 distance |
| 4 | **SpectralDiffuserCam** | Src(scene) -> C(diffuser) -> W(spectral_filter_array) -> D(CMOS) | Diffuser + spectral CFA |

Sizes: 128x128, 270x480, 512x512 (3)
Case Count: 4 x 3 x 4 x 5 = **240 per benchmark**

Data Sources:
- DiffuserCam DLMD (25K pairs) `WEB` https://waller-lab.github.io/LenslessLearning/dataset.html
- PhlatCam (10K measurements) `WEB` https://github.com/vboomi/PhlatCam
- Waller Lab 100K parallel dataset `WEB` https://waller-lab.github.io/parallel-lensless-dataset/
- LenslessPiCam toolkit `WEB` https://github.com/LCAV/LenslessPiCam

### 5.2 Panorama Multi-Focus Fusion
Variants (2): Multi-exposure fusion, Multi-focus fusion
Case Count: 2 x 3 x 4 x 5 = **120 per benchmark**

### 5.3 Coded Exposure / Flutter Shutter
Variants (2): Binary temporal code, Learned temporal code
Case Count: 2 x 3 x 4 x 5 = **120 per benchmark**

### 5.4 Event Camera / Dynamic Vision Sensor (DVS)
Variants (3): Standard DVS, DAVIS (combined frame+events), Color DVS
Case Count: 3 x 3 x 4 x 5 = **180 per benchmark**
Data Source: DAVIS dataset `WEB`

### 5.5 High Dynamic Range (HDR) Imaging
Variants (3): Multi-exposure HDR, Single-shot HDR (learned), Bracketed HDR
Case Count: 3 x 3 x 4 x 5 = **180 per benchmark**

**Category 5 Total: ~3,840**

---

## Category 6: Computational Optics (2 modalities)

### 6.1 Light Field Imaging

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **Plenoptic 1.0** | Src(scene) -> C(main_lens) -> S(microlens_array_125um) -> D(CMOS) | Microlens array before sensor |
| 2 | **Plenoptic 2.0** | Src(scene) -> C(main_lens) -> S(focused_microlens) -> D(CMOS) | Focused microlens array |
| 3 | **Camera Array** | [Src->C->D]_N, N=17x17 | Multiple cameras on gantry |

Sizes: 256x256x5x5, 512x512x9x9, 1024x1024x14x14 (3)
Case Count: 3 x 3 x 4 x 5 = **180 per benchmark**

Data Sources:
- Stanford Light Field Archive `WEB` http://lightfield.stanford.edu/
- EPFL Light Field Image Dataset `WEB` https://www.epfl.ch/labs/mmspg/downloads/epfl-light-field-image-dataset/
- Stanford Lytro Archive `WEB` http://lightfields.stanford.edu/LF2016.html

### 6.2 Integral Photography
Variants (2): Standard integral, Computational integral
Case Count: 2 x 3 x 4 x 5 = **120 per benchmark**

**Category 6 Total: ~1,536**

---

## Category 7: Neural Rendering (2 modalities)

### 7.1 Neural Radiance Fields (NeRF)

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **NeRF (original MLP)** | Src(scene) -> M(volume_density) -> P(ray_marching) -> D(camera) | 1.2M param MLP + volume rendering |
| 2 | **Mip-NeRF 360** | Same + cone_casting | 9M params; unbounded scenes |
| 3 | **Instant-NGP** | Same + hash_encoding | 5M params; seconds to train |
| 4 | **NeRF--** | Same but no_known_poses | Additional pose estimation |
| 5 | **Zip-NeRF** | Mip-NeRF + hash grid | State of art |

Sizes: 400x400, 800x800, 1920x1080 (3)
Views: 25, 50, 100, 200 (4)
Case Count: 5 x 3 x 4 x 4 x 5 = **1,200 per benchmark**

Data Sources:
- Blender Synthetic (8 objects, 800x800) `WEB` Google Drive
- LLFF (8 forward-facing scenes) `WEB` https://github.com/Fyusion/LLFF
- Tanks and Temples (14 outdoor) `WEB` https://www.tanksandtemples.org/
- Mip-NeRF 360 (9 scenes) `WEB` https://jonbarron.info/mipnerf360/
- DL3DV-10K (10,510 scenes, CVPR 2024) `WEB` https://github.com/DL3DV-10K/Dataset

### 7.2 3D Gaussian Splatting (3DGS)

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **3DGS (original)** | Src(scene) -> M(3D_Gaussians_100K-5M) -> P(differentiable_rasterizer) -> D(camera) | Explicit splats + tile-based rasterization |
| 2 | **2D Gaussian Splatting** | Same but disk_primitives | Better surface reconstruction |
| 3 | **Scaffold-GS** | Same + voxel_scaffolding | Compact; fewer Gaussians |
| 4 | **Mip-Splatting** | Same + 3D_smoothing | Anti-aliasing |
| 5 | **4DGS (Dynamic)** | Same + temporal_dimension | Dynamic scenes |

Case Count: 5 x 3 x 4 x 4 x 5 = **1,200 per benchmark**

Data Sources: Same as NeRF + Deep Blending `WEB` https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

**Category 7 Total: ~9,600**

---

## Category 8: Electron Microscopy (11 modalities)

### Per-Modality Variants and Cases

| # | Modality | Variants | Per-BM Cases | Data Source |
|---|----------|----------|-------------|-------------|
| 1 | SEM | 3 (SE, BSE, low-kV) | 120 | Generated `GEN` |
| 2 | TEM | 3 (bright-field, dark-field, HRTEM) | 120 | EMPIAR `WEB` https://www.ebi.ac.uk/empiar/ |
| 3 | Electron Tomography | 3 (TEM-tomo, STEM-tomo, cryo-ET) | 180 | EMPIAR `WEB` |
| 4 | STEM | 3 (HAADF, ABF, iDPC) | 120 | Generated `GEN` |
| 5 | 4D-STEM / Electron Diffraction | 2 (thin, thick) | 80 | Generated `GEN` |
| 6 | EBSD | 2 (standard, HR-EBSD) | 80 | Generated `GEN` |
| 7 | EELS | 2 (core-loss, low-loss) | 80 | Generated `GEN` |
| 8 | Electron Holography | 2 (off-axis, in-line) | 80 | Generated `GEN` |
| 9 | Cryo-ET | 3 (standard, cryo-FIB, sub-tomogram avg) | 120 | EMPIAR `WEB`, EMDB `WEB` |
| 10 | FIB-SEM | 2 (sequential, PFIB) | 80 | OpenOrganelle `WEB` https://openorganelle.janelia.org/ |
| 11 | EDX Mapping | 2 (spot, map) | 80 | Generated `GEN` |

**Category 8 Total: ~4,560**

---

## Category 9: Depth Imaging (5 modalities)

| # | Modality | Variants | Per-BM Cases | Data Source |
|---|----------|----------|-------------|-------------|
| 1 | ToF Camera | 3 (CW, pulsed, iToF) | 120 | NYU Depth V2 `WEB` |
| 2 | LiDAR | 3 (mechanical, solid-state, flash) | 120 | KITTI `WEB` https://www.cvlibs.net/datasets/kitti/ |
| 3 | Structured Light | 3 (binary, sinusoidal, speckle) | 120 | Middlebury stereo `SYN-WEB` |
| 4 | Photometric Stereo | 2 (Lambertian, non-Lambertian) | 80 | DiLiGenT `WEB` |
| 5 | Flash LiDAR | 2 (SPAD, APD array) | 80 | Generated `GEN` |

**Category 9 Total: ~2,080**

---

## Category 10: Remote Sensing (11 modalities)

| # | Modality | Variants | Per-BM Cases | Data Source |
|---|----------|----------|-------------|-------------|
| 1 | SAR | 3 (stripmap, spotlight, ScanSAR) | 120 | SpaceNet SAR `WEB` |
| 2 | Sonar | 3 (side-scan, multibeam, SAS) | 120 | Generated `GEN` |
| 3 | Hyperspectral Remote | 3 (pushbroom, whiskbroom, snapshot) | 120 | Indian Pines/Pavia Univ `WEB` |
| 4 | Multispectral Sat | 2 (Landsat, Sentinel-2) | 80 | Sentinel Hub `WEB` |
| 5 | GPR | 2 (pulsed, stepped-frequency) | 80 | Generated `GEN` |
| 6 | Weather Radar | 2 (single-pol, dual-pol) | 80 | NEXRAD `WEB` |
| 7 | Radio Interferometry | 2 (VLA, VLBI) | 80 | CASA simulation `WEB` |
| 8 | Passive Microwave | 2 (single-freq, multi-freq) | 80 | Generated `GEN` |
| 9 | InSAR | 2 (repeat-pass, tandem) | 80 | Copernicus data `WEB` |
| 10 | PolSAR | 2 (quad-pol, compact-pol) | 80 | PolSARpro `WEB` |
| 11 | Ocean Color | 2 (MODIS, VIIRS) | 80 | NASA Ocean Color `WEB` |

**Category 10 Total: ~3,920**

---

## Category 11: Industrial Inspection (10 modalities)

| # | Modality | Variants | Per-BM Cases |
|---|----------|----------|-------------|
| 1 | Industrial CT | 3 (micro-CT, macro-CT, in-line CT) | 120 |
| 2 | X-ray NDT | 2 (film, DR) | 80 |
| 3 | Ultrasonic Phased Array | 3 (TFM, FMC, TOFD) | 120 |
| 4 | Eddy Current | 2 (single-freq, multi-freq) | 80 |
| 5 | Active Thermography | 3 (pulsed, lock-in, step-heating) | 120 |
| 6 | Terahertz | 2 (TDS, CW) | 80 |
| 7 | Machine Vision | 3 (2D inspection, 3D profilometry, defect detect) | 120 |
| 8 | XRF Imaging | 2 (micro-XRF, mapping) | 80 |
| 9 | Shearography | 2 (temporal, spatial) | 80 |
| 10 | Scanning Acoustic Microscopy | 2 (reflection, transmission) | 80 |

**Category 11 Total: ~3,840**

---

## Category 12: Scientific Instrumentation (12 modalities)

| # | Modality | Variants | Per-BM Cases | Data Source |
|---|----------|----------|-------------|-------------|
| 1 | X-ray Crystallography | 3 (single-crystal, powder, Laue) | 120 | PDB `WEB` https://www.rcsb.org/ |
| 2 | SAXS | 2 (standard, GISAXS) | 80 | SASView `WEB` |
| 3 | MALDI MSI | 2 (TOF, FTICR) | 80 | Generated `GEN` |
| 4 | Atom Probe (APT) | 2 (voltage, laser) | 80 | Generated `GEN` |
| 5 | Cryo-EM SPA | 3 (standard, Volta phase plate, energy filter) | 120 | EMPIAR `WEB` |
| 6 | Neutron Tomography | 2 (thermal, cold) | 80 | Generated `GEN` |
| 7 | Proton Radiography | 2 (pencil beam, broad beam) | 80 | Generated `GEN` |
| 8 | Muon Tomography | 2 (scattering, absorption) | 80 | Generated `GEN` |
| 9 | WAXS | 2 (standard, GIWAXS) | 80 | Generated `GEN` |
| 10 | XRF Tomography | 2 (pencil beam, full-field) | 80 | Generated `GEN` |
| 11 | Neutron Diffraction | 2 (powder, single-crystal) | 80 | Generated `GEN` |
| 12 | Cathodoluminescence | 2 (SEM-CL, STEM-CL) | 80 | Generated `GEN` |

**Category 12 Total: ~4,160**

---

## Category 13: Broader Experimental Science (11 modalities)

| # | Modality | Variants | Per-BM Cases | Data Source |
|---|----------|----------|-------------|-------------|
| 1 | Adaptive Optics | 3 (Shack-Hartmann, pyramid, predictive) | 120 | Generated `GEN` |
| 2 | Seismic Tomography | 3 (body-wave, surface-wave, ambient noise) | 120 | IRIS `WEB` https://www.iris.edu/ |
| 3 | Gravitational Wave | 2 (ground-based, space-based) | 80 | LIGO Open Science `WEB` https://gwosc.org/ |
| 4 | Particle Calorimetry | 3 (EM, hadronic, dual-readout) | 120 | CERN Open Data `WEB` |
| 5 | Radio Aperture Synthesis | 3 (VLA, ALMA, LOFAR) | 120 | NRAO archive `WEB` |
| 6 | Acoustic Emission | 2 (resonant, broadband) | 80 | Generated `GEN` |
| 7 | MPI | 2 (system-matrix, x-space) | 80 | OpenMPIData `WEB` |
| 8 | EIT | 2 (static, time-difference) | 80 | EIDORS `WEB` |
| 9 | FWI | 3 (acoustic, elastic, visco-elastic) | 120 | SEG/EAGE models `WEB` |
| 10 | Ocean Acoustic Tomo | 2 (ray-based, wave-based) | 80 | Generated `GEN` |
| 11 | BLT | 2 (free-space, diffusion) | 80 | Generated `GEN` |

**Category 13 Total: ~4,320**

---

## Category 14: Spectroscopy & Spectral Imaging (8 modalities)

| # | Modality | Variants | Per-BM Cases |
|---|----------|----------|-------------|
| 1 | Raman Imaging | 3 (spontaneous, confocal Raman, tip-enhanced TERS) | 120 |
| 2 | CARS | 2 (broadband, multiplex) | 80 |
| 3 | SRS | 2 (single-frequency, hyperspectral) | 80 |
| 4 | FTIR Imaging | 3 (transmission, ATR, micro-FTIR) | 120 |
| 5 | LIBS Imaging | 2 (single-pulse, double-pulse) | 80 |
| 6 | Brillouin | 2 (spontaneous, stimulated) | 80 |
| 7 | SIMS | 2 (static, dynamic) | 80 |
| 8 | DESI MSI | 2 (standard, nano-DESI) | 80 |

**Category 14 Total: ~2,880**

---

## Category 15: Ultrafast Imaging (4 modalities)

| # | Modality | Variants | Per-BM Cases |
|---|----------|----------|-------------|
| 1 | Streak Camera | 3 (single-shot, synchroscan, FESCA) | 120 |
| 2 | Pump-Probe | 3 (degenerate, non-degenerate, broadband) | 120 |
| 3 | CUP | 3 (standard, T-CUP, mega-frame) | 120 |
| 4 | XFEL SFX | 3 (standard, time-resolved, serial Laue) | 120 |

**Category 15 Total: ~1,920**

---

## Category 16: Quantum Imaging (3 modalities)

| # | Modality | Variants | Per-BM Cases |
|---|----------|----------|-------------|
| 1 | Ghost Imaging | 3 (thermal, SPDC, computational) | 120 |
| 2 | Quantum Illumination | 2 (idler-signal, SU(1,1)) | 80 |
| 3 | Entangled Photon Microscopy | 2 (biphoton, NOON-state) | 80 |

**Category 16 Total: ~1,120**

---

## Category 17: Multi-Modal Fusion (6 modalities)

| # | Modality | Variants | Per-BM Cases | Data Source |
|---|----------|----------|-------------|-------------|
| 1 | PET/CT | 3 (sequential, integrated, TOF-PET/CT) | 180 | autoPET `WEB` |
| 2 | PET/MR | 3 (sequential, simultaneous, insert) | 180 | Generated `GEN` |
| 3 | SPECT/CT | 2 (parallel-hole, pinhole) | 120 | Generated `GEN` |
| 4 | US/MRI Fusion | 2 (rigid, deformable registration) | 120 | Generated `GEN` |
| 5 | CT + Fluorescence (FLIT) | 2 (sequential, simultaneous) | 120 | Generated `GEN` |
| 6 | CLEM | 3 (pre-embedding, post-embedding, correlative cryo) | 180 | Generated `GEN` |

**Category 17 Total: ~3,600**

---

## Category 18: Scanning Probe Microscopy (4 modalities)

| # | Modality | Variants | Per-BM Cases |
|---|----------|----------|-------------|
| 1 | AFM | 4 (contact, tapping, PeakForce, high-speed) | 160 |
| 2 | STM | 3 (constant-current, constant-height, STS) | 120 |
| 3 | NSOM | 3 (aperture, apertureless, scattering) | 120 |
| 4 | MFM | 2 (standard, quantitative) | 80 |

**Category 18 Total: ~1,920**

---

## Category 19: Astronomy & Space Imaging (4 modalities)

| # | Modality | Variants | Per-BM Cases | Data Source |
|---|----------|----------|-------------|-------------|
| 1 | Coronagraphy | 4 (Lyot, vortex, PIAA, starshade) | 240 | STScI archive `WEB` |
| 2 | Lucky Imaging | 2 (short-exposure selection, speckle) | 120 | Generated `GEN` |
| 3 | EHT Imaging | 2 (standard VLBI, space VLBI) | 120 | EHT Collaboration `WEB` https://eventhorizontelescope.org/ |
| 4 | Solar EUV/X-ray | 3 (SDO/AIA, STEREO, Hinode) | 180 | SDO archive `WEB` https://sdo.gsfc.nasa.gov/ |

**Category 19 Total: ~2,640**

---

## Grand Total (All 19 Categories)

| Category | Modalities | Approximate Total Cases |
|----------|-----------|------------------------|
| 1. Compressive Imaging | 4 | 22,608 |
| 2. Microscopy | 24 | 11,976 |
| 3. Medical Imaging | 37 | 21,252 |
| 4. Coherent Imaging | 5 | 6,700 |
| 5. Computational Photography | 5 | 3,840 |
| 6. Computational Optics | 2 | 1,536 |
| 7. Neural Rendering | 2 | 9,600 |
| 8. Electron Microscopy | 11 | 4,560 |
| 9. Depth Imaging | 5 | 2,080 |
| 10. Remote Sensing | 11 | 3,920 |
| 11. Industrial Inspection | 10 | 3,840 |
| 12. Scientific Instrumentation | 12 | 4,160 |
| 13. Broader Experimental Science | 11 | 4,320 |
| 14. Spectroscopy & Spectral | 8 | 2,880 |
| 15. Ultrafast Imaging | 4 | 1,920 |
| 16. Quantum Imaging | 3 | 1,120 |
| 17. Multi-Modal Fusion | 6 | 3,600 |
| 18. Scanning Probe | 4 | 1,920 |
| 19. Astronomy & Space | 4 | 2,640 |
| **GRAND TOTAL** | **168** | **~114,472** |

> With full combinatorial expansion including all image sizes, noise levels, compression ratios,
> and mismatch levels, the total exceeds **297,920** test instances across all benchmarks.
