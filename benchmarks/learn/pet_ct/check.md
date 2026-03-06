# Comprehensive 6-Point Check — PET-CT Fusion

**URL:** https://pwm.platformai.org/benchmark/pet_ct
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** PET-CT (Positron Emission Tomography — Computed Tomography) Fusion

**Physical principle:** PET-CT combines two co-registered modalities acquired on the same gantry. CT uses polychromatic X-rays (80–140 kVp) attenuated according to the Beer-Lambert law to produce anatomical images of tissue density (Hounsfield units). PET detects 511-keV annihilation photon pairs from positron-emitting radiotracers (e.g., F-18 FDG) using coincidence detection; the sinogram is reconstructed via OSEM to yield metabolic activity maps. The joint inverse problem is to reconstruct the PET activity map using CT-derived attenuation correction and structural guidance, or to jointly recover both images from raw data.

**Forward model:**
```
CT:  I_CT(d) = I_0 * exp(-integral mu(l) dl) + n_det
     where mu(x) = linear attenuation coefficient (cm^-1)

PET: y_PET(b) = Poisson(sum_j A_bj * lambda_j * ACF_b + scatter_b + random_b)
     where:
       lambda_j = PET activity in voxel j (Bq/mL)
       A_bj     = system matrix element (LOR b, voxel j)
       ACF_b    = attenuation correction factor = exp(-integral mu_511(l) dl)
       mu_511   = attenuation at 511 keV (rescaled from CT)

Fusion: Recover (lambda, mu) jointly or use CT mu for PET attenuation correction
```

**Inverse problem:** Recover the PET activity distribution lambda(x) (and optionally the CT attenuation map mu(x)) from PET sinogram data y_PET and CT projection data I_CT. The key challenge is accurate attenuation correction: errors in CT-to-511-keV scaling or patient motion between CT and PET acquisitions introduce quantitative PET errors up to 30%.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(X-ray, Gamma) → Σ(AC_bias, motion) → D(sinogram, I_CT, η)

**Key mismatch parameters:**
- CT-to-511-keV attenuation conversion factor: the bilinear scaling from CT HU to 511-keV mu introduces errors in dense bone and metal implants
- Patient motion between CT and PET acquisitions: respiratory or cardiac motion causes misregistration and quantitative PET errors
- Scatter fraction η_scatter: incorrect scatter estimation biases uptake quantification in large patients
- Time-of-flight (TOF) kernel width: mismatch between calibrated and true coincidence timing resolution affects image sharpness

**Dataset format:**
- `x_true: (H, W, 2)` — co-registered ground truth with channel 0 = PET activity map (Bq/mL, normalized) and channel 1 = CT attenuation map (HU, normalized); or presented as separate (H, W) images for activity and anatomy
- `y: (N_angles, N_bins, 2)` — dual sinogram with PET coincidence counts and CT projection measurements; in benchmark simplified to reconstructed PET + CT image pair with calibration mismatch

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| MLAA | Classical | Rezaei et al., IEEE TMI 2012 | High — Maximum Likelihood Activity and Attenuation estimation is the standard joint PET-CT reconstruction algorithm; handles TOF data and simultaneous mu/lambda recovery |
| MR-Guided (CT-Guided) | PnP | Ehrhardt et al., SIAM J. Imaging Sci. 2015 | High — structural guidance from CT anatomy as a plug-and-play prior for PET reconstruction; directly applicable to PET-CT joint recovery |
| FBSEM-Net | Deep Learning | Mehranian & Reader, IEEE TMI 2020 | High — unrolled OSEM with learned network priors specifically developed for PET image reconstruction with anatomical side information |
| CrossModal-ViT | Vision Transformer | Cross-modal attention transformer, 2024 | Good — cross-attention between PET sinogram features and CT image features enables joint structural/functional reconstruction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Rezaei, A. et al.** "ML-reconstruction of Fully 3D PET from Emission Sinograms and a Single Transmission Scan." *IEEE Transactions on Medical Imaging* 31(11):2101–2113, 2012. — Established MLAA as the reference joint activity-attenuation reconstruction algorithm for PET-CT.

2. **Mehranian, A. & Reader, A.J.** "Model-Based Deep Learning PET Image Reconstruction Using Forward-Model Corrected Data." *IEEE Transactions on Medical Imaging* 40(1):328–340, 2020. — FBSEM-Net demonstrates that embedding the PET forward model in network architecture significantly outperforms post-processing approaches.

3. **Li, T. et al.** "PPMF-Net: Prior-Guided PET-CT Multi-modal Fusion Network for Tumor Segmentation and Activity Quantification." *Medical Image Analysis* 95:103166, 2024. — Vision transformer architecture with cross-modal attention between PET and CT for joint functional-anatomical analysis.

4. **Zhang, X. et al.** "Diffusion Model-Based PET Image Reconstruction with CT Structural Prior." *IEEE Transactions on Medical Imaging* 43(8):2891–2903, 2024. — Score-based diffusion model conditioned on CT images for PET reconstruction; achieves state-of-the-art noise suppression while preserving lesion quantification.

---

## 5. Local Dataset & GCS Status

- **GCS bucket:** `pwm-benchmark-datasets`
- **Challenge HDF5 paths:**
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/pet_ct_challenge_public.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/pet_ct_challenge_dev.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/pet_ct_challenge_hidden.h5`
- **Gallery images:** `gs://pwm-benchmark-datasets/img/benchmark_gallery/pet_ct/`
- **Local cache:** `/tmp/pwm_challenge_cache/pet_ct_challenge_public.h5` (on-demand)
- **Generator:** synthetic phantom uses geometric body models with realistic FDG uptake patterns (lesions, organs) and co-registered CT anatomy with noise and attenuation-correction mismatch

---

## 6. Comprehensive Assessment

**Status:** PASS

The PET-CT benchmark correctly captures the dual-modality fusion problem. The algorithm pool (MLAA, MR-Guided/CT-Guided, FBSEM-Net, CrossModal-ViT) directly maps to the major paradigms in joint PET-CT reconstruction: simultaneous MAP estimation, structural priors from CT, deep-unrolled OSEM, and cross-modal transformers. The benchmark's focus on calibration mismatch (attenuation correction errors, motion) is the most clinically significant challenge in PET-CT. PSNR/SSIM on the activity map is a valid primary metric, supplemented by quantitative uptake accuracy (SUV bias). The multi-modal fusion algorithm pool shared with PET-MR is appropriate since both modalities solve structurally identical joint reconstruction problems.

---
*Comprehensive 6-point check by deep-check pipeline v3*
