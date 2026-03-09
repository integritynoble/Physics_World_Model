# Comprehensive 6-Point Check — Fluorescence-Guided CT (CT Fluorescence)

**URL:** https://pwm.platformai.org/benchmark/ct_fluorescence
**Check Date:** 2026-03-09
**Status:** NEEDS_WORK

---

## 1. Physics & Forward Model

**Modality:** Fluorescence-Guided CT (CT Fluorescence)

**Physical principle:** CT fluorescence (also known as fluorescence computed tomography or fluorescence-guided CT) combines X-ray CT structural imaging with co-registered fluorescence optical imaging in a dual-modality system. The X-ray channel provides anatomical context and tissue attenuation maps, while the fluorescence channel detects localized fluorophore (probe) distributions within the same field of view. The fluorescence signal obeys diffuse optical transport through tissue, making this a hybrid radiological + optical inverse problem.

**Forward model:**
```
p_CT(s, θ)   = ∫ μ(r) dl + n_CT             (X-ray CT: Radon transform)
y_FL(r_d)    = ∫∫ G(r_d, r) * q(r) * φ(r) dV + n_FL   (fluorescence: diffuse transport)

where:
  μ(r)        — X-ray linear attenuation coefficient map
  q(r)        — fluorophore concentration map (target)
  φ(r)        — excitation fluence field (from diffusion equation)
  G(r_d, r)   — Green's function from fluorophore position r to detector r_d
  n_CT, n_FL  — Poisson noise (CT) and detector noise (fluorescence)
```

**Inverse problem:** Jointly or sequentially recover (1) the X-ray attenuation map `μ` from sinogram data and (2) the fluorophore distribution `q` from surface fluorescence measurements, optionally using `μ` as anatomical prior for the optical inverse problem.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(tissue anatomy + probe distribution) → F(X-ray Radon + optical diffusion) → D(detector array + optical CCD)

**Key mismatch parameters:**
- `optical_scattering_coeff`: Reduced scattering coefficient μ_s'; nominal 1.0 mm⁻¹, perturbed 0.5–2.0 mm⁻¹
- `ct_views`: Number of X-ray projection angles; nominal 180, perturbed 30–90
- `fluorophore_depth`: Depth of fluorophore inclusion; nominal 10 mm, perturbed 5–25 mm
- `background_autofluorescence`: Ratio of background to target fluorescence; nominal 0.05, perturbed 0.0–0.3

**Dataset format:**
- `x_true: (H, W)` — ground-truth fluorophore concentration map (co-registered, 256×256)
- `y: (N_views, N_det)` — CT sinogram and surface fluorescence measurements (stacked or separate channels)

**Public datasets:**
- Virtual Photonics toolkit (vts.usc.edu, open-source) — Monte Carlo photon transport code for generating CT-fluorescence training and test datasets; standard in biomedical optics
- NIRFAST simulation package (dartmouth.edu, open-source) — FEM-based near-infrared fluorescence tomography simulation for generating validated CT-fluorescence phantom data
- Simulated datasets from Arridge group (UCL) and Ntziachristos group (TU Munich) — open-access supporting data in published papers

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| FBP + Tikhonov Optical Reconstruction | Classical | Arridge, Inverse Problems 15:R41 (1999) | Mandatory baseline — sequential FBP for CT Radon inversion, Tikhonov-regularized Born inversion for fluorescence DOT; THE standard dual-modality reconstruction pipeline |
| CT-guided DOT (anatomically-constrained) | Model-based | Hyde et al., IEEE Trans. Med. Imaging 29:365 (2009) | CT segmentation defines spatial priors for fluorescence DOT reconstruction; required model-based baseline |
| Dual-modality deep reconstruction (CT-Fluo-Net) | Deep Learning | Gao et al., IEEE TNNLS 2021; extended 2022 | CNN jointly processing CT and fluorescence to reconstruct probe distribution; required DL baseline |
| Physics-informed NeRF for fluorescence CT | Deep Learning | Zhu et al., Nature 555:487 (2018) methodology adapted | Implicit neural representation incorporating diffusion equation as physics constraint |

**ACTION REQUIRED:** Source Virtual Photonics or NIRFAST simulation datasets. Register FBP + Tikhonov optical reconstruction (Arridge 1999, Inverse Problems) as mandatory classical baseline in YAML. Register CT-Fluo-Net (2022) as required DL baseline in YAML.

---

## 4. Literature & State of the Art (2024–2025)

1. **Cao, X. et al. (2024)** "Anatomically-guided fluorescence molecular tomography using deep learning," *Biomedical Optics Express* 15(2):789–804 — CT-prior-guided U-Net reconstruction reduces fluorescence localization error by 40%.
2. **Chen, J. et al. (2024)** "Simultaneous X-ray CT and fluorescence tomography on a clinical scanner," *J. Biomed. Opt.* 29(3):036001 — first clinical-grade dual-modality system with co-registered acquisitions on the same gantry.
3. **Liu, X. et al. (2024)** "Self-supervised multimodal reconstruction for CT-fluorescence imaging," *Phys. Med. Biol.* 69(8):085006 — self-supervised approach without paired training data; cross-modal consistency loss for joint optimization.
4. **Zhang, W. et al. (2025)** "Diffusion model-regularized fluorescence tomography reconstruction guided by CT anatomy," *Medical Physics* — score-based diffusion prior conditioned on CT segmentation for fluorophore recovery in deep tissue.

---

## 5. Local Dataset & GCS Status

**No challenge data ingested.** Challenge data to be generated from Virtual Photonics or NIRFAST simulation tools.

**Recommended public data sources:**
- Virtual Photonics toolkit (vts.usc.edu, open-source) — Monte Carlo photon transport simulation; standard in biomedical optics for CT-fluorescence dataset generation
- NIRFAST (dartmouth.edu, open-source) — FEM-based near-infrared fluorescence tomography simulation for validated phantom data generation
- Published simulation datasets (Arridge group UCL, Ntziachristos group TU Munich) — open-access supporting materials in published work

**GCS datasets (planned):**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ct_fluorescence_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ct_fluorescence_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ct_fluorescence_challenge_hidden.h5`

**Gallery images:** To be served from `gs://pwm-benchmark-datasets/img/benchmark_gallery/ct_fluorescence/`.

---

## 6. Comprehensive Assessment

**Status:** NEEDS_WORK

The CT fluorescence benchmark correctly captures the dual-modality inverse problem combining Radon-transform X-ray CT with diffuse optical fluorescence tomography. Algorithm routing spans sequential FBP+Tikhonov baselines, anatomically-constrained CT-guided DOT, and modern deep-learning multimodal networks, appropriately reflecting the hybrid nature of this imaging modality. The mismatch parameters on optical scattering, fluorophore depth, and background autofluorescence are the dominant physical parameters governing fluorescence reconstruction quality in real tissue. No challenge data has been ingested. Virtual Photonics or NIRFAST simulation tools must be used to generate datasets (very limited open experimental data exists for this modality).

**Outstanding items:**
1. No challenge data — generate using Virtual Photonics toolkit (vts.usc.edu) or NIRFAST (dartmouth.edu); very limited open experimental data exists.
2. Register FBP + Tikhonov optical reconstruction (Arridge 1999, Inverse Problems 15:R41) as mandatory classical baseline in YAML.
3. Register CT-Fluo-Net (2022) as required DL baseline in YAML.

---
*Comprehensive 6-point check by deep-check pipeline v4*
