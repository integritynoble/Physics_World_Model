# PWM Implementation Plan: Standard Datasets and Algorithms

This document outlines the implementation plan for integrating standard benchmark datasets and classical reconstruction algorithms for all 17 imaging modalities. The goal is to ensure reproducible results that match published benchmarks.

---

## Executive Summary

| Phase | Scope | Priority |
|-------|-------|----------|
| Phase 1 | Core modalities (Widefield, CT, MRI, SPC) | High |
| Phase 2 | Microscopy (SIM, Confocal, Lightsheet) | High |
| Phase 3 | Compressive (CASSI, CACTI) | Medium |
| Phase 4 | Advanced (Ptychography, Holography, NeRF, 3DGS) | Medium |

---

## Phase 1: Core Modalities

### 1.1 Widefield Microscopy

**Selected Dataset:** DeconvolutionLab2 Test Images
- Source: http://bigwww.epfl.ch/deconvolution/
- Images: Ely, CElegans, Sperm
- Resolution: 256×256, 512×512
- PSF: Measured or synthetic Gaussian

**Selected Algorithm:** Richardson-Lucy Deconvolution
- Reference: Richardson (1972), Lucy (1974)
- Expected PSNR: 25-30 dB (after 50 iterations)
- Published benchmark: DeconvolutionLab2 paper

**Implementation Tasks:**
```
[ ] Download DeconvolutionLab2 test images
[x] Implement Richardson-Lucy solver (richardson_lucy.py)
[ ] Validate against DeconvolutionLab2 results
[x] Add to PWM solver portfolio
```

**Code Location:** `pwm_core/recon/richardson_lucy.py`

**Validation Metrics:**
| Image | Published PSNR | Target PSNR |
|-------|----------------|-------------|
| Ely | 28.5 dB | ≥28.0 dB |
| CElegans | 26.2 dB | ≥26.0 dB |

---

### 1.2 Widefield Low-Dose

**Selected Dataset:** Fluorescence Microscopy Deconvolution (FMD)
- Source: https://github.com/zj-dong/FMD
- Contains: Confocal, Two-photon, Widefield
- Noise levels: Multiple SNR levels

**Selected Algorithm:** VST + BM3D
- Reference: Mäkitalo & Foi (2012)
- Variance Stabilizing Transform for Poisson noise
- Expected PSNR: 28-32 dB

**Implementation Tasks:**
```
[ ] Download FMD dataset
[x] Implement Anscombe VST (in pnp.py)
[x] Integrate BM3D denoising (in pnp.py)
[ ] Validate on low-SNR images
```

---

### 1.3 CT (Computed Tomography)

**Selected Dataset:** LoDoPaB-CT
- Source: https://zenodo.org/record/3384092
- Size: 35,820 training, 3,522 validation, 3,553 test
- Resolution: 362×362
- Views: Various (sparse to full)

**Selected Algorithm:** FBP + SART
- FBP Reference: Feldkamp (1984)
- SART Reference: Andersen & Kak (1984)
- Expected PSNR: 30-35 dB (full view), 25-30 dB (sparse)

**Implementation Tasks:**
```
[ ] Download LoDoPaB-CT dataset
[x] Implement FBP with Ram-Lak filter (ct_solvers.py)
[x] Implement SART iterative solver (ct_solvers.py)
[ ] Validate against published results
[ ] Integrate with ASTRA toolbox (optional)
```

**Code Location:** `pwm_core/recon/ct_solvers.py`

**Validation Metrics (LoDoPaB-CT):**
| Method | 1000 views | 128 views | 64 views |
|--------|------------|-----------|----------|
| FBP | 35.2 dB | 28.1 dB | 24.3 dB |
| SART | 36.1 dB | 30.5 dB | 27.2 dB |
| Target | ≥35 dB | ≥28 dB | ≥24 dB |

---

### 1.4 MRI (Magnetic Resonance Imaging)

**Selected Dataset:** fastMRI
- Source: https://fastmri.org/
- Subsets: Knee (single-coil, multi-coil), Brain
- Resolution: 320×320
- Accelerations: 4x, 8x

**Selected Algorithm:** ESPIRiT + Compressed Sensing
- Reference: Uecker et al. (2014)
- L1-wavelet regularization
- Expected PSNR: 32-38 dB (4x), 28-34 dB (8x)

**Implementation Tasks:**
```
[ ] Download fastMRI knee dataset (single-coil)
[x] Implement ESPIRiT coil sensitivity estimation (mri_solvers.py)
[x] Implement L1-wavelet CS reconstruction (mri_solvers.py)
[ ] Validate against fastMRI leaderboard
[ ] Optional: Integrate BART toolbox
```

**Code Location:** `pwm_core/recon/mri_solvers.py`

**Validation Metrics (fastMRI Knee 4x):**
| Method | Published SSIM | Published PSNR | Target |
|--------|----------------|----------------|--------|
| Zero-filled | 0.65 | 28.5 dB | Baseline |
| L1-ESPIRiT | 0.78 | 34.2 dB | ≥0.75 SSIM |
| VarNet | 0.89 | 38.1 dB | Future |

---

### 1.5 SPC (Single-Pixel Camera)

**Selected Dataset:** Set11
- Source: Standard CS benchmark (included in most papers)
- Images: 11 grayscale images
- Resolution: 256×256
- Widely used in: Kulkarni 2016, Zhang 2018

**Selected Algorithm:** ISTA-Net+ (or TVAL3 as classical baseline)
- TVAL3 Reference: Li et al. (2013)
- ISTA-Net Reference: Zhang & Ghanem (2018)
- Measurement: Hadamard patterns

**Sampling Rates:** 1%, 4%, 10%, 25%, 50%

**Implementation Tasks:**
```
[x] Prepare Set11 dataset (256×256) (data/loaders/set11.py)
[x] Implement Hadamard measurement matrix (data/loaders/set11.py)
[x] Implement TVAL3 solver (cs_solvers.py)
[ ] Validate at standard sampling rates
[ ] Compare with published ISTA-Net results
```

**Code Location:** `pwm_core/recon/cs_solvers.py`

**Validation Metrics (Set11, Hadamard):**
| Sampling Rate | TVAL3 PSNR | ISTA-Net PSNR | Target |
|---------------|------------|---------------|--------|
| 1% | 18.5 dB | 22.1 dB | ≥18 dB |
| 4% | 23.2 dB | 27.8 dB | ≥23 dB |
| 10% | 27.1 dB | 32.4 dB | ≥27 dB |
| 25% | 32.5 dB | 37.2 dB | ≥32 dB |
| 50% | 38.2 dB | 41.5 dB | ≥38 dB |

---

## Phase 2: Microscopy Modalities

### 2.1 Confocal Microscopy

**Selected Dataset:** BioSR / Cell Image Library
- BioSR: https://zenodo.org/record/5654567
- Cell Image Library: http://www.cellimagelibrary.org/

**Selected Algorithm:** 3D Richardson-Lucy
- Reference: McNally et al. (1999)
- Handles axial elongation of PSF

**Implementation Tasks:**
```
[ ] Download BioSR confocal subset
[ ] Implement 3D Richardson-Lucy
[ ] Handle anisotropic PSF
[ ] Validate on standard test volumes
```

---

### 2.2 SIM (Structured Illumination Microscopy)

**Selected Dataset:** BioSR / fairSIM Test Data
- BioSR: Paired widefield/SIM ground truth
- fairSIM: https://github.com/fairSIM/fairSIM-testdata
- Patterns: 9 images (3 angles × 3 phases)

**Selected Algorithm:** Wiener-SIM
- Reference: Gustafsson (2000), Heintzmann & Cremer (1999)
- Standard reconstruction in fairSIM

**Implementation Tasks:**
```
[ ] Download fairSIM test data
[x] Implement Wiener-SIM reconstruction (sim_solver.py)
[x] Validate pattern parameter estimation (sim_solver.py)
[ ] Compare with fairSIM output
```

**Code Location:** `pwm_core/recon/sim_solver.py`

**Validation Metrics:**
| Dataset | fairSIM PSNR | Target |
|---------|--------------|--------|
| Actin | 28.5 dB | ≥28 dB |
| Tubulin | 27.2 dB | ≥27 dB |

---

### 2.3 Light-Sheet Microscopy

**Selected Dataset:** OpenSPIM Sample Data
- Source: https://openspim.org/
- Multi-view acquisitions
- 3D volumes with stripe artifacts

**Selected Algorithm:** Multi-View Fusion + Stripe Removal
- Reference: Preibisch et al. (2010)
- Fiji/BigStitcher implementation as reference

**Implementation Tasks:**
```
[ ] Download OpenSPIM sample data
[ ] Implement stripe removal filter
[ ] Implement multi-view registration
[ ] Implement weighted fusion
```

---

## Phase 3: Compressive Imaging

### 3.1 CASSI (Coded Aperture Spectral Imaging)

**Selected Dataset:** KAIST + ICVL (simulation)
- KAIST: http://vclab.kaist.ac.kr/siggraphasia2017p1/
- ICVL: http://icvl.cs.bgu.ac.il/hyperspectral/
- Resolution: 256×256×28 bands

**Selected Algorithm:** GAP-TV
- Reference: Yuan (2016)
- Code: https://github.com/yuanxy92/GAP-TV
- Widely used baseline for CASSI

**Implementation Tasks:**
```
[ ] Download KAIST/ICVL datasets
[x] Implement GAP-TV solver (gap_tv.py)
[x] Implement CASSI forward model (dispersion) (physics/cassi_operator.py)
[ ] Validate on 10 benchmark scenes
```

**Code Location:** `pwm_core/recon/gap_tv.py`

**Validation Metrics (KAIST, 256×256×28):**
| Scene | GAP-TV PSNR | GAP-TV SSIM | Target |
|-------|-------------|-------------|--------|
| Scene 1 | 32.5 dB | 0.92 | ≥32 dB |
| Scene 2 | 31.8 dB | 0.91 | ≥31 dB |
| Average | 32.1 dB | 0.915 | ≥31.5 dB |

---

### 3.2 CACTI (Video SCI)

**Selected Dataset:** 6 Benchmark Videos
- Source: https://github.com/mq0829/DUN-3DUnet
- Videos: Kobe, Traffic, Runner, Drop, Crash, Aerial
- Resolution: 256×256×8 frames

**Selected Algorithm:** GAP-TV (video version)
- Reference: Yuan (2016)
- Same algorithm family as CASSI
- Baseline for all CACTI papers

**Implementation Tasks:**
```
[x] Download 6 benchmark videos (data/loaders/cacti_bench.py - synthetic)
[x] Implement shifting mask model (data/loaders/cacti_bench.py)
[x] Implement GAP-TV for video (gap_tv.py)
[ ] Validate on all 6 videos
```

**Validation Metrics (256×256×8):**
| Video | GAP-TV PSNR | GAP-TV SSIM | Target |
|-------|-------------|-------------|--------|
| Kobe | 26.8 dB | 0.84 | ≥26 dB |
| Traffic | 24.5 dB | 0.81 | ≥24 dB |
| Runner | 29.2 dB | 0.89 | ≥29 dB |
| Drop | 34.1 dB | 0.94 | ≥34 dB |
| Crash | 25.3 dB | 0.83 | ≥25 dB |
| Aerial | 25.8 dB | 0.82 | ≥25 dB |
| Average | 27.6 dB | 0.855 | ≥27 dB |

---

### 3.3 Lensless / DiffuserCam

**Selected Dataset:** DiffuserCam Lensless Mirflickr
- Source: https://waller-lab.github.io/LenslessLearning/
- Real captured lensless images
- Measured PSF included

**Selected Algorithm:** ADMM
- Reference: Boyd et al. (2011)
- Standard for DiffuserCam reconstruction

**Implementation Tasks:**
```
[ ] Download DiffuserCam dataset
[ ] Implement ADMM with TV prior
[ ] Use measured PSF
[ ] Validate on test images
```

**Validation Metrics:**
| Method | PSNR | SSIM | Target |
|--------|------|------|--------|
| ADMM (100 iter) | 24.5 dB | 0.78 | ≥24 dB |
| ADMM (500 iter) | 26.2 dB | 0.82 | ≥26 dB |

---

## Phase 4: Advanced Modalities

### 4.1 Ptychography

**Selected Dataset:** Synthetic Ptychography Data
- Generate using known object + probe
- Overlap: 60-80%
- Positions: 64-256 scan points

**Selected Algorithm:** ePIE (extended Ptychographic Iterative Engine)
- Reference: Maiden & Rodenburg (2009)
- Standard algorithm for ptychography

**Implementation Tasks:**
```
[x] Implement synthetic data generator (physics/ptychography_operator.py)
[x] Implement ePIE algorithm (ptychography_solver.py)
[ ] Validate on synthetic data (known ground truth)
[ ] Test probe recovery
```

**Code Location:** `pwm_core/recon/ptychography_solver.py`

---

### 4.2 Holography

**Selected Dataset:** Simulated Holography (DIV2K as object)
- Use DIV2K for simulation
- Off-axis holography setup
- Known propagation distance

**Selected Algorithm:** Angular Spectrum Method
- Reference: Goodman (2005)
- Standard for digital holography

**Implementation Tasks:**
```
[x] Implement off-axis hologram simulation (physics/holography_operator.py)
[x] Implement angular spectrum propagation (holography_solver.py)
[x] Implement phase unwrapping (holography_solver.py)
[ ] Validate on simulated data
```

---

### 4.3 NeRF (Neural Radiance Fields)

**Selected Dataset:** Synthetic-NeRF (Blender)
- Source: https://github.com/bmild/nerf
- 8 synthetic scenes
- 100 training views, 200 test views

**Selected Algorithm:** Instant-NGP (or vanilla NeRF)
- Reference: Mildenhall et al. (2020), Müller et al. (2022)
- Hash encoding for fast training

**Implementation Tasks:**
```
[ ] Download Synthetic-NeRF dataset
[ ] Integrate NeRF framework (nerfstudio)
[ ] Validate on Lego, Chair, etc.
[ ] Report PSNR/SSIM metrics
```

**Validation Metrics (Synthetic-NeRF):**
| Scene | NeRF PSNR | Instant-NGP PSNR | Target |
|-------|-----------|------------------|--------|
| Lego | 32.5 dB | 35.2 dB | ≥32 dB |
| Chair | 33.0 dB | 35.8 dB | ≥33 dB |
| Average | 31.0 dB | 33.5 dB | ≥31 dB |

---

### 4.4 3D Gaussian Splatting

**Selected Dataset:** Mip-NeRF 360
- Source: https://github.com/google-research/multinerf
- Indoor/outdoor scenes
- Unbounded scenes

**Selected Algorithm:** 3D Gaussian Splatting
- Reference: Kerbl et al. (2023)
- State-of-the-art for novel view synthesis

**Implementation Tasks:**
```
[ ] Download Mip-NeRF 360 dataset
[ ] Integrate 3DGS codebase
[ ] Validate on benchmark scenes
```

---

## Implementation Schedule

### Week 1-2: Dataset Infrastructure
```
[x] Create dataset download scripts (data/download.py)
[x] Implement data loaders for each format (data/loaders/)
[x] Set up benchmark evaluation pipeline (benchmarks/run_all.py)
[ ] Create results logging system
```

### Week 3-4: Phase 1 Algorithms
```
[x] Richardson-Lucy (Widefield) - richardson_lucy.py
[x] FBP + SART (CT) - ct_solvers.py
[x] ESPIRiT + L1-wavelet (MRI) - mri_solvers.py
[x] TVAL3 / ISTA (SPC) - cs_solvers.py
```

### Week 5-6: Phase 2 Algorithms
```
[x] Wiener-SIM - sim_solver.py
[x] 3D Richardson-Lucy - richardson_lucy.py
[ ] Stripe removal + Multi-view fusion
```

### Week 7-8: Phase 3 Algorithms
```
[x] GAP-TV (CASSI) - gap_tv.py
[x] GAP-TV (CACTI) - gap_tv.py
[ ] ADMM (Lensless) - partially in pnp.py
```

### Week 9-10: Phase 4 Algorithms
```
[x] ePIE (Ptychography) - ptychography_solver.py
[x] Angular Spectrum (Holography) - holography_solver.py
[ ] NeRF integration (framework only)
[ ] 3DGS integration (framework only)
```

---

## File Structure

```
packages/pwm_core/
├── pwm_core/
│   ├── recon/
│   │   ├── __init__.py
│   │   ├── richardson_lucy.py      # Widefield
│   │   ├── ct_solvers.py           # FBP, SART
│   │   ├── mri_solvers.py          # ESPIRiT, CS
│   │   ├── cs_solvers.py           # TVAL3, ISTA
│   │   ├── sim_solver.py           # Wiener-SIM
│   │   ├── gap_tv.py               # CASSI, CACTI
│   │   ├── ptychography_solver.py  # ePIE
│   │   ├── holography_solver.py    # Angular spectrum
│   │   └── pnp.py                  # PnP methods
│   └── data/
│       ├── __init__.py
│       ├── download.py             # Dataset download
│       └── loaders/
│           ├── lodopab.py          # CT
│           ├── fastmri.py          # MRI
│           ├── set11.py            # SPC
│           ├── kaist.py            # CASSI
│           └── cacti_bench.py      # CACTI
├── examples/
│   └── data/
│       ├── ct/
│       ├── mri/
│       ├── spc/
│       ├── cassi/
│       └── cacti/
└── benchmarks/
    ├── run_all.py
    ├── evaluate_ct.py
    ├── evaluate_mri.py
    └── results/
```

---

## Validation Criteria

For each modality, we consider implementation successful when:

1. **PSNR Match**: Within 0.5 dB of published results
2. **SSIM Match**: Within 0.02 of published results
3. **Visual Quality**: No visible artifacts compared to reference
4. **Runtime**: Reasonable execution time (specified per algorithm)

---

## Summary: Selected Datasets and Algorithms

| # | Modality | Dataset | Algorithm | Expected PSNR |
|---|----------|---------|-----------|---------------|
| 1 | Widefield | DeconvolutionLab2 | Richardson-Lucy | 25-30 dB |
| 2 | Widefield Low-Dose | FMD | VST + BM3D | 28-32 dB |
| 3 | Confocal Live-Cell | BioSR | Richardson-Lucy | 24-28 dB |
| 4 | Confocal 3D | BioSR | 3D Richardson-Lucy | 24-28 dB |
| 5 | SIM | fairSIM | Wiener-SIM | 27-30 dB |
| 6 | CASSI | KAIST | GAP-TV | 30-34 dB |
| 7 | SPC | Set11 | TVAL3 / ISTA-Net | 25-40 dB* |
| 8 | CACTI | 6 Videos | GAP-TV | 25-34 dB |
| 9 | Lensless | DiffuserCam | ADMM | 24-28 dB |
| 10 | Lightsheet | OpenSPIM | Multi-View Fusion | N/A (3D) |
| 11 | CT | LoDoPaB-CT | FBP / SART | 25-36 dB |
| 12 | MRI | fastMRI | ESPIRiT + L1 | 30-38 dB |
| 13 | Ptychography | Synthetic | ePIE | 30-40 dB |
| 14 | Holography | Simulated | Angular Spectrum | 30-40 dB |
| 15 | NeRF | Synthetic-NeRF | NeRF / Instant-NGP | 31-36 dB |
| 16 | 3DGS | Mip-NeRF 360 | 3D Gaussian Splatting | 25-32 dB |
| 17 | Matrix | Custom | CG / ADMM | Problem-specific |

*SPC PSNR depends heavily on sampling rate (1%-50%)

---

## Next Steps

1. **Review this plan** and confirm dataset/algorithm choices
2. **Prioritize modalities** based on user needs
3. **Begin Phase 1 implementation** with CT and MRI (most standard benchmarks)
4. **Set up CI/CD** for automated benchmark testing

---

## References

1. Richardson, W.H. (1972). "Bayesian-based iterative method of image restoration"
2. Lucy, L.B. (1974). "An iterative technique for the rectification of observed distributions"
3. Gustafsson, M.G.L. (2000). "Surpassing the lateral resolution limit by a factor of two using SIM"
4. Yuan, X. (2016). "Generalized alternating projection based total variation minimization for compressive sensing"
5. Uecker, M. et al. (2014). "ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI"
6. Maiden, A.M. & Rodenburg, J.M. (2009). "An improved ptychographical phase retrieval algorithm"
7. Mildenhall, B. et al. (2020). "NeRF: Representing Scenes as Neural Radiance Fields"
8. Kerbl, B. et al. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
