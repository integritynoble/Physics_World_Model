# Benchmark QA Check — ceus

**URL:** https://pwm.platformai.org/benchmark/ceus
**Check Date:** 2026-03-03 (manual deep review)
**Status:** PASS (with observations)

---

## 1. Platform Page Review

### Page Content (fetched from https://pwm.platformai.org/benchmark/ceus)

The benchmark page describes **Contrast-Enhanced Ultrasound (CEUS)** as a blind
reconstruction challenge within the Physics World Model framework. The task
requires recovering original ultrasound signals from degraded measurements under
unknown parameter mismatches.

**Forward Model Architecture:** Propagation --> Rotation --> Detector (P --> R --> D)

**Mismatch Parameters (from platform page):**

| Parameter                      | Range          | Unit         |
|-------------------------------|----------------|--------------|
| Bubble concentration          | -1.0 to 2.0   | relative     |
| Nonlinear harmonic extraction | -2.0 to 4.0   | dimensionless|
| Motion between frames         | -1.0 to 2.0   | mm           |

**Note:** The platform page shows narrower, signed ranges (e.g., -1.0 to 2.0)
while the local config uses wider, unsigned ranges (e.g., 0.1 to 5.0). This
discrepancy may reflect different parameterisation conventions (additive offset
vs. multiplicative factor), but should be verified for consistency.

**Scoring Methodology (composite metric):**
- 40% Peak Signal-to-Noise Ratio (PSNR, normalised)
- 40% Structural Similarity Index (SSIM)
- 20% Consistency term: (1 - ||y - H_hat * x_hat|| / ||y||)

The local config lists only `psnr` and `ssim` without the 20% consistency term.
The platform page's composite metric is more informative -- the local config
should be updated to reflect this 40/40/20 weighting.

**Evaluation Tiers:**
- Public (3 scenes): full ground truth + mismatch values provided
- Dev (3 scenes): blind evaluation, no ground truth
- Hidden (3 scenes): server-side containerised evaluation

**Input/Output Format:** HDF5 files for measurements (y), forward operator (H),
parameter ranges (input) and reconstructed signals (x_hat) with corrected spec
parameters (output).

### Leaderboard (from platform page)

| Rank | Method               | Overall | Public PSNR/SSIM  | Dev PSNR/SSIM    | Hidden PSNR/SSIM |
|------|----------------------|---------|-------------------|------------------|------------------|
| 1    | MU-Net + gradient    | 0.671   | 31.72 / 0.940     | 25.61 / 0.821    | 23.3 / 0.743     |
| 2    | ABLE + gradient      | 0.623   | 30.53 / 0.925     | 22.87 / 0.726    | 20.66 / 0.630    |
| 3    | PnP-ADMM + gradient  | 0.622   | 25.64 / 0.822     | 24.55 / 0.788    | 22.35 / 0.705    |

**Observation:** PnP-ADMM ranks third overall but shows the smallest public-to-
hidden PSNR gap (3.29 dB vs. 8.42 dB for MU-Net and 9.87 dB for ABLE). This
suggests PnP-ADMM is the most robust to increasing mismatch severity, while the
deep-learning methods overfit to the public tier conditions.

---

## 2. Literature Review

### CEUS Super-Resolution and Reconstruction (2024-2025)

**Ultrasound Localization Microscopy (ULM):**
CEUS enables super-resolution imaging by tracking individual microbubbles
(1-5 um diameter) across frames. Deep learning has transformed this field:

- **Deep-ULM** (van Sloun et al.): Fully convolutional neural network for
  super-resolution image reconstruction from dense, overlapping microbubble
  signals. Bypasses classical detect-localise-track pipelines by learning an
  end-to-end mapping from CEUS frames to sub-diffraction-limit images.

- **LOCA-ULM** (2024, Nature Communications): Context-aware deep learning
  achieving 97.8% microbubble detection accuracy at high concentrations.
  Demonstrates that deep learning can handle the dense-bubble regime where
  classical methods fail due to overlapping point-spread functions.
  Source: https://www.nature.com/articles/s41467-024-47154-2

- **Sub-pixel accuracy improvements** (2024, PMC): Supervised and self-
  supervised deep learning for sub-pixel microbubble localisation, reducing
  location error by ~75% compared to conventional blind deconvolution, with
  60x speed improvement. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10774911/

**Physics-Informed Approaches:**
- Physics-informed neural networks (PINNs) are being applied to ultrasound
  inverse problems, combining forward-model priors with neural network
  flexibility. Model-based deep learning reduces computational demand while
  maintaining reconstruction quality (Frontiers in Physics, 2024).
  Source: https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1398393/full

- Non-invasive medical digital twins using physics-informed self-supervised
  learning (NeurIPS 2024) demonstrate the broader trend of embedding physical
  constraints into neural network architectures for medical imaging.
  Source: https://proceedings.neurips.cc/paper_files/paper/2024/file/0b081a44ed0b8c0c4aa6bd886a60bea4-Paper-Conference.pdf

**Optimisation-Based Tracking:**
- Analytic optimisation-based microbubble tracking treats temporal pairing as
  a bubble-set registration problem, using PSF similarity and physical motion
  constraints (arXiv:2209.10057).
  Source: https://arxiv.org/abs/2209.10057

**Clinical Deep Learning on CEUS:**
- 3D CNN with temporal/channel attention for hepatocellular carcinoma analysis
  on CEUS video (Springer, 2024). ResNet-18 architectures for lesion
  classification. These are downstream tasks (diagnosis) rather than
  reconstruction, but demonstrate the clinical value of high-quality CEUS
  reconstruction.

### Relevance to PWM Benchmark

The PWM benchmark focuses on the *beamforming and reconstruction* step (image
formation from raw RF data under mismatch), which is upstream of the ULM
localisation pipeline. The literature shows two convergent trends:

1. **Unrolled optimisation** (PnP-ADMM family): physics-aware, robust to
   model mismatch, lower peak performance but graceful degradation.
2. **End-to-end deep learning** (MU-Net, ABLE): higher peak PSNR on matched
   data, but larger performance drops under mismatch.

The leaderboard results confirm this pattern exactly: MU-Net leads on public
tier but PnP-ADMM has the most stable cross-tier performance.

---

## 3. Local Dataset Status

**Command:** `ls datasets/benchmark/ceus 2>/dev/null`
**Result:** Directory does not exist (exit code 2).

No local CEUS dataset is present. The benchmark configuration specifies:
- **Primary data source:** `us_kidney` from CAMUS challenge
  (https://www.creatis.insa-lyon.fr/Challenge/camus/)
- **Fallback:** `generated` using `shepp_logan` synthetic generator
- **Citation:** Leclerc et al., TMI 2019
- **License:** Research use

The expanded config also lists a `ceus_generated` synthetic source. The data
priority order is: web > experimental > synthetic_web > generated.

**Action needed:** Download CAMUS data or generate synthetic data before running
the benchmark locally:
```bash
python benchmarks/runners/run_expanded.py --modality ceus --solver traditional_cpu
```

---

## 4. Configuration and Code Review

### Config Files

| File                                          | Size   | Status |
|-----------------------------------------------|--------|--------|
| `benchmarks/configs/ceus.yaml`                | present | OK     |
| `benchmarks/expanded_configs/ceus_expanded.yaml` | present | OK  |
| `docs/modality_benchmarks/ceus.md`            | present | OK     |

### Key Configuration Details

| Property              | Value                               |
|-----------------------|-------------------------------------|
| Modality ID           | `ceus`                              |
| Display Name          | Contrast-Enhanced Ultrasound (CEUS) |
| Category              | Medical Imaging                     |
| Carrier               | Acoustic                            |
| DAG                   | P --> R --> D                        |
| Maturity              | M0                                  |
| Tier                  | A                                   |
| Forward Model Type    | nonlinear_operator                  |
| Category Module       | medical_ct_radon                    |
| Default Solver        | contrast_specific                   |
| Operator ID           | ceus                                |
| Has Dedicated Operator| true                                |
| Graph Template        | ceus_graph_v1                       |
| Image Shape (x)       | [64, 64]                            |
| Measurement Shape (y) | [64, 64]                            |

### Registered Solvers

| Tier             | Name     | Module                    | Function          | Params | GPU |
|------------------|----------|---------------------------|-------------------|--------|-----|
| traditional_cpu  | FBP      | pwm_core.recon.fbp        | run_fbp           | 0      | No  |
| best_quality     | DL-Recon | pwm_core.recon.dl_recon   | dl_reconstruct    | 5M     | Yes |

### Algorithms from Catalog (via modify_plan.md)

| # | Algorithm | Type          | Source                          |
|---|-----------|---------------|---------------------------------|
| 1 | DAS       | Classical     | Analytical baseline             |
| 2 | PnP-ADMM  | PnP           | Goudarzi et al., 2020           |
| 3 | ABLE      | Deep Learning | Luijten et al., IEEE TMI 2020   |
| 4 | MU-Net    | Deep Learning | Hyun et al., IEEE TUFFC 2022    |

All citations verified as correct. Routing path: (medical, Acoustic) ->
medical_ultrasound pool.

### Expanded Config: Benchmark Scale

| Phase | Cases |
|-------|-------|
| B1    | 12    |
| B2    | 40    |
| B3    | 40    |
| B4    | 40    |
| **Grand Total** | **132** |

Noise levels: clean (60 dB), low (40 dB), medium (30 dB), high (20 dB).
Image sizes: small [256, 256], standard [512, 512].

### Learning Materials

All 5 curriculum files plus README present and verified:

| File                          | Size     | Status |
|-------------------------------|----------|--------|
| README.md                     | 1,455 B  | OK     |
| 01_physics_fundamentals.md    | 2,077 B  | OK     |
| 02_forward_model.md           | 2,700 B  | OK     |
| 03_reconstruction_algorithms.md| 2,018 B | OK     |
| 04_pwm_benchmark.md           | 2,420 B  | OK     |
| 05_hands_on_tutorial.md       | 3,476 B  | OK     |

---

## 5. Issues and Observations

### Discrepancies

| # | Severity | Description |
|---|----------|-------------|
| 1 | WARNING  | **Mismatch range discrepancy:** Platform page shows signed ranges (-1.0 to 2.0, -2.0 to 4.0, -1.0 to 2.0) while local configs show unsigned ranges (0.1-5.0, 0.0-10.0, 0.0-5.0). These should be reconciled. |
| 2 | WARNING  | **Scoring formula mismatch:** Platform page uses a 40/40/20 composite (PSNR + SSIM + consistency), but local config only lists psnr and ssim without weights or the consistency term. |
| 3 | INFO     | **Leaderboard name mismatch:** Previous check.md listed "US-Transformer + gradient" and "PnP-DRUNet + gradient" but the actual algorithm catalog has MU-Net and PnP-ADMM. The modify_plan.md confirms this stale naming issue. |
| 4 | INFO     | **Category module questionable:** The `category_module` is set to `medical_ct_radon` (Radon/projection-based), which is designed for CT-like modalities. CEUS is fundamentally a pulse-echo acoustic modality, not projection-based. A dedicated `medical_ultrasound` or `ceus_specific` module would be more physically accurate. |
| 5 | INFO     | **Config image shape vs. expanded config:** Base config uses [64, 64] for both x and y shapes, while expanded config offers [256, 256] (small) and [512, 512] (standard). The 64x64 base shape is very low resolution for CEUS imaging. |
| 6 | INFO     | **No local dataset:** `datasets/benchmark/ceus/` does not exist. Data must be downloaded or generated before benchmarking. |
| 7 | INFO     | **Solver gap:** Only 2 solver tiers registered (traditional_cpu, best_quality). The algorithm catalog has 4 methods (DAS, PnP-ADMM, ABLE, MU-Net), but only FBP and DL-Recon are wired into the solver config. The `famous_dl` and `small_gpu` tiers referenced in the algorithm selection guide are not populated. |
| 8 | INFO     | **Wavelength/Energy Range placeholder:** 01_physics_fundamentals.md lists "0 - 0 nm" which is a template placeholder. Medical CEUS typically uses 1-10 MHz centre frequencies (wavelengths ~0.15-1.5 mm in tissue). |

### Positive Findings

- All platform pages load correctly (main, public, dev, compete, contribute)
- HDF5 data files verified on GCS for public and dev tiers
- Forward model reference present and linked to graph template
- Complete 5-part learning curriculum with ~2 hours estimated reading time
- Algorithm pool correctly routed via (medical, Acoustic) -> medical_ultrasound
- All 4 algorithm citations verified as accurate
- 132 total benchmark cases across B1-B4 phases with 4 noise levels

---

## 6. Comprehensive Summary

### Overall Assessment: PASS with minor issues

The CEUS benchmark is structurally complete and functional. The platform page
is live, data files are accessible on GCS, the algorithm pool contains four
real, correctly-cited ultrasound reconstruction methods, and the 5-part learning
curriculum is fully populated.

### Maturity: M0 (Template)

The benchmark is at the earliest maturity level. To advance to M1:
- Reconcile mismatch parameter ranges between platform and config
- Add the consistency term to local scoring or document why it differs
- Replace the `medical_ct_radon` category module with ultrasound-specific physics
- Fill in the wavelength/energy range placeholder
- Populate the `famous_dl` and `small_gpu` solver tiers
- Download or generate CEUS-specific training data locally

### Key Technical Insight

The leaderboard reveals the classic robustness-vs-peak-performance trade-off:
MU-Net achieves the highest overall score (0.671) but suffers 8.42 dB PSNR
degradation from public to hidden tier. PnP-ADMM scores lower overall (0.622)
but degrades only 3.29 dB, demonstrating superior robustness to model mismatch.
This is precisely the kind of insight the PWM benchmark is designed to surface.

### Recommended Priority Actions

1. **HIGH:** Reconcile mismatch ranges between platform page and local config
2. **HIGH:** Add 40/40/20 composite scoring formula to local metrics config
3. **MEDIUM:** Replace `medical_ct_radon` with ultrasound-appropriate physics module
4. **MEDIUM:** Wire all 4 catalog algorithms (DAS, PnP-ADMM, ABLE, MU-Net) into solver config tiers
5. **LOW:** Fix wavelength placeholder in physics fundamentals
6. **LOW:** Create `datasets/benchmark/ceus/` with downloaded or generated data

---

**Tags:** `#ceus` `#medical-imaging` `#ultrasound` `#acoustic` `#nonlinear` `#M0` `#benchmark-qa` `#deep-review`

---
*Generated by manual deep review on 2026-03-03.*

Sources:
- [Context-aware deep learning for high-concentration microbubble ULM (Nature Communications, 2024)](https://www.nature.com/articles/s41467-024-47154-2)
- [Sub-pixel accuracy in ULM using deep learning (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10774911/)
- [Deep learning in medical ultrasound imaging survey (Frontiers in Physics, 2024)](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1398393/full)
- [Non-invasive medical digital twins with PINNs (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/0b081a44ed0b8c0c4aa6bd886a60bea4-Paper-Conference.pdf)
- [Analytic optimization-based microbubble tracking (arXiv)](https://arxiv.org/abs/2209.10057)
- [Deep-ULM: Super-resolution ULM through deep learning (arXiv)](https://arxiv.org/pdf/1804.07661)
- [Deep learning microbubble localisation for ULM (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9116497/)