# Comprehensive Benchmark QA Check — CBCT

**URL:** https://pwm.platformai.org/benchmark/cbct
**HTTP Status:** 200
**Check Date:** 2026-03-03 (comprehensive 6-point review)
**Reviewer:** Local server (automated + manual deep analysis)

---

## Table of Contents

1. [Benchmark Page Errors](#1-benchmark-page-errors)
2. [Local Dataset Inspection](#2-local-dataset-inspection)
3. [Public Dataset Source Assessment](#3-public-dataset-source-assessment)
4. [Algorithm Coverage Assessment](#4-algorithm-coverage-assessment)
5. [Improvement Suggestions](#5-improvement-suggestions)
6. [Action Items](#6-action-items)

---

## 1. Benchmark Page Errors

### Summary

| Severity | Count |
|----------|-------|
| HIGH     | 5     |
| MEDIUM   | 8     |
| LOW      | 5     |

### HIGH Severity

**H1. Mismatch parameter discrepancy between webpage and local data**
- Webpage shows 4 parameters: center_offset, source_dist, cone_angle, detector_tilt
- Local spec.json has 6 parameters: source_offset_x, source_offset_z, detector_tilt, detector_shift_u, beam_hardening, scatter_fraction
- Parameter names differ, and webpage is missing beam_hardening and scatter_fraction
**Fix:** Sync webpage mismatch table with actual spec.json (6 parameters).

**H2. Sample count mismatch**
- Webpage: "3 scenes" per tier (9 total)
- Local README: 10 public, 20 dev, 20 hidden (50 total)
**Fix:** Update webpage to reflect actual sample counts.

**H3. PSNR_norm undefined in scoring formula**
Scoring formula uses "PSNR_norm" without defining normalization.
**Fix:** Define PSNR_norm = (PSNR - PSNR_min) / (PSNR_max - PSNR_min) with explicit bounds.

**H4. No HDF5 files exist for any tier**
- All 3 tiers only contain README.md, spec.json, true_spec.json
- No actual projection data or ground-truth volumes
- Dataset has NOT been built yet
**Fix:** Run `simulate_phantoms.py` and dataset builder to generate H5 files.

**H5. Forward model on webpage oversimplified**
- Webpage: R(theta) -> Pi(cone) -> D(g, eta)
- README: Full model with gain, offset, Poisson+Gaussian noise, 6 mismatch knobs
- Missing: scatter model, beam hardening polynomial
**Fix:** Update forward model section to include all 6 mismatch effects.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Geometry parameters not on webpage (SID=600mm, SDD=1200mm, detector 512x512, 0.8mm pitch) |
| M2 | N_views varies across tiers (256 public, 128-512 dev/hidden) but webpage shows fixed views |
| M3 | Procedural phantom complexity (fBm, Worley, TPMS, L-systems) not documented on webpage |
| M4 | Hidden tier rank inversion: PnP-DRUNet 3rd in dev but 4th in hidden, FBP overtakes it |
| M5 | Missing references: TransCT, FBPConvNet, NAF not cited on webpage |
| M6 | Attenuation scale table not on webpage (air=0, cortical bone=0.8, metal=1.0) |
| M7 | Detector non-uniformity (gain/offset) mentioned in README but not in webpage forward model |
| M8 | Volume size 256^3 not specified on webpage |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Placeholder links: /benchmark/cbct/compete, /benchmark/cbct/contribute |
| L2 | No gallery images showing reconstruction comparisons |
| L3 | No alt-text on any images |
| L4 | Spec DAG has no figure caption |
| L5 | SSIM parameters (window size, data_range) not specified |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | Files Present | H5 Data? | Size |
|------|---------------|----------|------|
| Public | README.md, spec.json, true_spec.json | **NO** | metadata only |
| Dev | README.md, spec.json, true_spec.json | **NO** | metadata only |
| Hidden | README.md, spec.json, true_spec.json | **NO** | metadata only |

**CRITICAL: No HDF5 dataset files exist. The CBCT dataset has NOT been built.**

### Spec Verification (from spec.json)

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| source_offset_x | -2.0 | 2.0 | mm |
| source_offset_z | -1.5 | 1.5 | mm |
| detector_tilt | -0.5 | 0.5 | deg |
| detector_shift_u | -3.0 | 3.0 | px |
| beam_hardening | 0.0 | 0.15 | (unitless) |
| scatter_fraction | 0.0 | 0.10 | (unitless) |

Note: source_offset and detector_tilt are signed (bidirectional), so negative values are physically correct.

### True Spec (Public Tier)

| Parameter | Value |
|-----------|-------|
| source_offset_x | 0.80 mm |
| source_offset_z | 0.50 mm |
| detector_tilt | 0.15 deg |
| detector_shift_u | 1.20 px |
| beam_hardening | 0.06 |
| scatter_fraction | 0.04 |

### Phantom Generator

`simulate_phantoms.py` (~1200 lines) exists and includes:
- 10 dev recipes (anatomy-inspired): head, thorax, abdomen, extremity, dental, pelvis, shoulder, knee, spine, hand
- 10 hidden recipes (adversarial): trabecular micro, multi-metal, vascular tree, lung, fractal membrane, gyroid scaffold, dental metal, cardiac, multi-contrast, reaction-diffusion
- Multi-scale fBm + Worley noise, TPMS surfaces, L-system vascular trees, reaction-diffusion patterns

### Dataset Integrity Assessment: **FAIL** — No H5 files exist, dataset must be built

---

## 3. Public Dataset Source Assessment

### Planned Sources (from README)

| # | Dataset | Anatomy | Year | Citations | License | Assessment |
|---|---------|---------|------|-----------|---------|------------|
| 00 | AAPM Low-Dose CT (Mayo) | Abdomen | 2017 | ~800 | Research | EXCELLENT - Gold standard clinical CT |
| 01 | AAPM Low-Dose CT (Mayo) | Chest | 2017 | ~800 | Research | EXCELLENT - Gold standard |
| 02 | LoDoPaB-CT | Chest | 2021 | ~200 | CC BY 4.0 | EXCELLENT - Standard low-dose benchmark |
| 03 | 2DeteCT | Industrial | 2023 | ~50 | Open | GOOD - Recent, diverse geometry |
| 04 | Helsinki Tomography (HTC) | Acrylic disc | 2022 | ~100 | Open | GOOD - Community challenge |
| 05 | LIDC-IDRI | Lung (nodule) | 2011 | ~3,000 | CC BY 3.0 | EXCELLENT - Largest lung CT |
| 06 | Walnut CT (CWI) | Walnut | 2019 | ~150 | CC BY 4.0 | GOOD - Designed for ML, cone-beam |
| 07 | CWI Bamboo CT | Bamboo | 2021 | ~50 | Open | GOOD - Cone-beam micro-CT |
| 08 | FIPS Open Data | Head phantom | 2022 | ~30 | Open | GOOD - Known geometry, calibrated |
| 09 | Apple CT | Apple | 2020 | ~20 | Open | FAIR - Niche, low citation count |

### Overall Source Quality: **EXCELLENT**

- 10 diverse sources spanning clinical, industrial, and calibration phantoms
- AAPM Mayo, LoDoPaB-CT, and LIDC-IDRI are field standards (>5,000 combined citations)
- Good mix of medical (5), industrial (2), calibration (1), and biological (2) objects
- Cone-beam native data from Walnut CT and Bamboo CT adds authenticity

### Protection Assessment

| Tier | Protection Level | Method |
|------|-----------------|--------|
| Public | Open (by design) | Real data from published datasets |
| Dev | EXCELLENT | Procedural phantoms with 10 unique recipes + secret seeds |
| Hidden | EXCELLENT | Adversarial recipes (trabecular micro, multi-metal, etc.) + secret seeds |

---

## 4. Algorithm Coverage Assessment

### Currently Tested (Webpage Leaderboard)

| # | Algorithm | Type | Notes |
|---|-----------|------|-------|
| 1 | TransCT + gradient | Transformer-based DL | Top performer |
| 2 | FBPConvNet + gradient | CNN post-processing | Classic DL baseline |
| 3 | PnP-DRUNet + gradient | Plug-and-play denoiser | Degrades on hidden |
| 4 | FBP + gradient | Analytical (FDK) | Classical baseline |

### PWM Solver Registry

| Solver | Status |
|--------|--------|
| FDK / FBP | Registered (`ct_solvers.run_fbp`) |

**Only 1 solver registered for CBCT. Major gap.**

### Missing Famous/Recent Algorithms (MUST ADD)

| Priority | Algorithm | Year | Citation | Why Important |
|----------|-----------|------|----------|---------------|
| **CRITICAL** | FDK (Feldkamp-Davis-Kress) | 1984 | ~7,000 citations | THE foundational cone-beam algorithm |
| **CRITICAL** | SART / SIRT | 1984/2001 | ~3,000 citations | Standard iterative algebraic methods |
| **CRITICAL** | ASD-POCS (TV-regularized) | 2008 | ~2,500 citations | Gold standard iterative recon (Sidky & Pan) |
| **HIGH** | CGLS (Conjugate Gradient LS) | Classical | ~1,000+ | Fast iterative, no regularization |
| **HIGH** | NAF (Neural Attenuation Fields) | 2022 | ~200 citations (MICCAI) | NeRF-based CBCT recon, novel paradigm |
| **HIGH** | LEARN / iCT-Net | 2018/2020 | ~400 citations | Learned iterative reconstruction |
| **HIGH** | U-Net post-processing | 2017 | ~thousands | Standard DL post-processing baseline |
| **MEDIUM** | 3D Gaussian splatting for CT | 2024 | New | Emerging approach from MICCAI 2024 |
| **MEDIUM** | SwinIR + NAG | 2024 | ICASSP challenge winner | State-of-the-art for CBCT challenge 2024 |
| **MEDIUM** | FACT (meta-learned NAF) | 2024 | ~30 | Fast and accurate sparse-view CBCT |
| **MEDIUM** | Diffusion-based CBCT | 2024 | ~50 | Score-based priors for cone-beam |
| **LOW** | CycleGAN CBCT-to-CT | 2020 | ~500 | Unsupervised domain adaptation |
| **LOW** | rho-NeRF | 2024 | New | Attenuation-prior NeRF for CT |

### Algorithm Gap Analysis

| Category | Have | Missing | Gap |
|----------|------|---------|-----|
| Analytical | FBP | FDK (proper cone-beam) | 1 |
| Iterative algebraic | -- | SART, SIRT, CGLS | 3 |
| TV-regularized | -- | ASD-POCS, FISTA-TV | 2 |
| CNN post-processing | FBPConvNet | U-Net, LEARN, iCT-Net | 3 |
| Transformer | TransCT | SwinIR+NAG | 1 |
| NeRF/implicit | -- | NAF, FACT, 3D Gaussians, rho-NeRF | 4 |
| Plug-and-play | PnP-DRUNet | PnP-BM3D | 1 |
| Generative | -- | Diffusion-based, CycleGAN | 2 |

**Total gap: 17 algorithms missing across 8 categories**

---

## 5. Improvement Suggestions

### 5.1 Dataset Improvements

1. **BUILD THE DATASET (CRITICAL)**
   - No H5 files exist for any tier
   - Run `simulate_phantoms.py` + dataset builder to generate actual data
   - Without data, the benchmark is metadata-only

2. **Public tier: Download and process real data**
   - AAPM Mayo: requires TCIA account (free), DICOM format
   - LoDoPaB-CT: Zenodo record 3384092, direct download
   - Walnut CT: Zenodo, cone-beam native format
   - LIDC-IDRI: TCIA, DICOM, well-documented API

3. **Increase public samples from 10 to 15-20**
   - Add more AAPM Mayo slices (abdomen, pelvis, head)
   - Add LIDC-IDRI cases with different nodule types
   - Add Jaw CT dataset (dental CBCT-specific)

4. **Dev/Hidden: Verify phantom generator produces realistic volumes**
   - Run the 1200-line generator on dev seeds 0-19
   - Visual QA: check that head_cranial looks like a head, torso_thorax like a thorax
   - Quantitative QA: histogram should match clinical CT distribution

5. **Add variable N_views to public tier**
   - Currently all public at N=256
   - Add some at N=128 and N=64 for sparse-view testing
   - This is more clinically relevant (dose reduction)

6. **Add motion artifacts to hidden tier**
   - Real CBCT suffers from patient motion during long acquisition
   - Add simulated rigid/non-rigid motion as a hidden-tier challenge

### 5.2 Algorithm Testing Improvements

7. **Add SART/SIRT iterative baseline immediately**
   - These are the minimum expected iterative methods
   - Widely available in ASTRA Toolbox (Python)
   - No training required

8. **Add ASD-POCS (Sidky & Pan, 2008)**
   - Gold standard for sparse-view CT
   - Total variation regularization, well-understood convergence
   - Reference implementation available

9. **Add NAF (Neural Attenuation Fields)**
   - NeRF-based approach specifically designed for CBCT
   - MICCAI 2022 paper, growing adoption
   - Particularly strong for sparse-view

10. **Add ICASSP 2024 challenge winner (SwinIR + NAG)**
    - State-of-the-art for low-dose 3D CBCT
    - Combines sinogram enhancement + image enhancement
    - Available: https://arxiv.org/abs/2406.08048

11. **Run all algorithms on all 3 tiers consistently**
    - Current leaderboard: 4 algorithms x 3 tiers
    - Target: 8-10 algorithms x 3 tiers

### 5.3 Benchmark Infrastructure Improvements

12. **Sync webpage with local data**
    - Fix sample counts (10/20/20 not 3/3/3)
    - Fix mismatch parameter names and count (6 not 4)
    - Add geometry table (SID, SDD, detector size, voxel size)
    - Add attenuation scale table

13. **Add cone-beam-specific metrics**
    - Standard PSNR/SSIM may not capture cone-beam artifacts well
    - Consider: streak artifact metrics, metal artifact reduction metrics
    - Slice-by-slice vs volumetric evaluation

14. **Document N_views distribution**
    - Dev: 40% N=256, 30% N=512, 30% N=128
    - Hidden: 40% N=128, 30% N=256, 30% N=512
    - This critical detail is missing from webpage

15. **Add ASTRA Toolbox integration**
    - ASTRA is the standard GPU-accelerated cone-beam projector
    - Required for proper FDK/SART/SIRT implementation
    - Should be a dependency

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Build H5 dataset files for all 3 tiers | Dataset team | TODO |
| CRITICAL | Download public tier source data (AAPM, LoDoPaB, Walnut, etc.) | Dataset team | TODO |
| CRITICAL | Sync webpage sample counts (10/20/20) | Main server | TODO |
| CRITICAL | Sync webpage mismatch parameters (6 knobs, correct names) | Main server | TODO |
| CRITICAL | Add SART/SIRT to solver registry and leaderboard | Algorithm team | TODO |
| HIGH | Add FDK (proper cone-beam) to solver registry | Algorithm team | TODO |
| HIGH | Add ASD-POCS iterative reconstruction | Algorithm team | TODO |
| HIGH | Add NAF (Neural Attenuation Fields) | Algorithm team | TODO |
| HIGH | Add geometry parameters to webpage | Main server | TODO |
| HIGH | Define PSNR_norm formula on webpage | Main server | TODO |
| MEDIUM | Add SwinIR+NAG (ICASSP 2024 winner) | Algorithm team | TODO |
| MEDIUM | Add FACT (meta-learned NAF) | Algorithm team | TODO |
| MEDIUM | Add missing references (TransCT, FBPConvNet, NAF) | Main server | TODO |
| MEDIUM | Add motion artifacts to hidden tier | Dataset team | TODO |
| MEDIUM | Add cone-beam-specific artifact metrics | Metrics team | TODO |
| LOW | Add ASTRA Toolbox as dependency | Infrastructure | TODO |
| LOW | Add gallery reconstruction comparisons | Main server | TODO |
| LOW | Fix placeholder links (/compete, /contribute) | Main server | TODO |

---

## Appendix: Key References

- Feldkamp, L.A., Davis, L.C., Kress, J.W. "Practical cone-beam algorithm." JOSA A 1.6 (1984): 612-619.
- Andersen, A.H., Kak, A.C. "Simultaneous algebraic reconstruction technique (SART)." Ultrasonic Imaging 6.1 (1984): 81-94.
- Sidky, E.Y., Pan, X. "Image reconstruction in circular cone-beam CT by constrained, total-variation minimization." PMB 53.17 (2008): 4777.
- Jin, K.H., et al. "Deep convolutional neural network for inverse problems in imaging." IEEE TIP 26.9 (2017): 4509-4522.
- McCollough, C.H., et al. "Low-dose CT for the detection and classification of metastatic liver lesions." Med. Phys. 44.10 (2017): e339-e352.
- Leuschner, J., et al. "LoDoPaB-CT, a benchmark dataset for low-dose CT." Sci. Data 8 (2021): 109.
- Zha, R., et al. "NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction." MICCAI (2022).
- Coban, S.B., et al. "2DeteCT: A large 2D expandable CT dataset." Sci. Data 10 (2023).
- ICASSP 2024. "3D CBCT Challenge: Improved Cone Beam CT Reconstruction Using SwinIR-Based Enhancement." arXiv:2406.08048.
- Zang, G., et al. "Learning 3D Gaussians for Extremely Sparse-View CBCT." MICCAI (2024).

---

*Comprehensive 6-point review on 2026-03-03. Covers: page errors, local dataset verification, source quality, algorithm coverage, improvement suggestions, and action items.*
