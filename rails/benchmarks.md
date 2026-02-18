# PWM Benchmarks — SolveEverything.org Rail for Computational Imaging

> Reference: [Solve Everything: A Blueprint for the Next Decade](https://solveeverything.org/)
> by Dr. Alexander D. Wissner-Gross and Dr. Peter H. Diamandis

PWM implements the SolveEverything 10-gear abundance engine as a **rail for
computational imaging calibration**. This document collects every benchmark
that has been executed, the metrics used, and how they map to the
SolveEverything framework.

---

## 1. SolveEverything Metric Mapping

| SolveEverything Metric | PWM Translation | Definition | Units |
|------------------------|-----------------|-----------|-------|
| RoCS (Return on Cognitive Spend) | **RoIC** (Return on Imaging Compute) | dB recovered per GPU-hour of calibration | dB / GPU-hr |
| Spec-to-Artifact Score | OperatorGraph compilation rate | % of ExperimentSpecs that compile without manual intervention | % |
| TtP (Time-to-Property) | Time-to-Calibration | Elapsed time from raw measurement to validated corrected reconstruction | sec / scene |
| DR-AIS (Decision Records) | **DR-IS** (Decision Records for Imaging Systems) | Immutable audit trail: timestamp, action, evidence, Triad gate, confidence, compute consumed, SHA-256 | structured record |
| LG/H (Learning Gain per Hour) | Calibration learning rate | How quickly a new modality reaches rho >= 0.80 given documented SOPs | modalities / month |
| Counterfactual Packs | **cfpacks** | Adversarial evaluation sets per modality (probe + hidden splits, 7 Red Team categories) | scenarios / pack |

---

## 2. Core Outcome Metrics (Gear 2)

Every PWM benchmark reports these three mechanically verifiable metrics.

| Metric | Formula | Target (L3) | Target (L5) |
|--------|---------|-------------|-------------|
| **Recovery ratio (rho)** | (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II) | >= 0.80 | >= 0.95 |
| **Oracle gap** | PSNR_I - PSNR_III | <= 2 dB | <= 0.5 dB |
| **RoIC** | (PSNR_III - PSNR_II) / GPU-hours | monotonically improving | commoditized |

### 4-Scenario Evaluation Protocol (Gear 1: LIP-Arena)

| Scenario | Measurement Operator | Reconstruction Operator | Purpose |
|----------|---------------------|------------------------|---------|
| I (Ideal) | True H | True H | Oracle upper bound |
| II (Assumed) | True H | Nominal H_nom | Mismatch-impact baseline |
| III (Corrected) | True H | Calibrated H_hat | Calibration benefit |
| IV (Oracle Mask) | True H | Partial oracle | Partial upper bound |

---

## 3. CASSI Benchmark (PWMI-CASSI)

**Modality:** Coded Aperture Snapshot Spectral Imaging (SD-CASSI)
**Dataset:** 10 KAIST scenes (256 x 256 x 28 spectral bands)
**Mask:** TSA_simu_data/mask.mat (256 x 256 binary coded aperture)
**Mismatch model:** 5 parameters (mask_dx, mask_dy, mask_theta, disp_a1, disp_alpha)
**Noise model:** Poisson-Gaussian (alpha=100000, sigma=0.01)
**Calibration:** Algorithm 1 (hierarchical beam search) + Algorithm 2 (GPU gradient refinement)

### 3.1 Injected Mismatch Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| mask_dx | 1.5 | px |
| mask_dy | 1.0 | px |
| mask_theta | 0.3 | deg |
| disp_a1 | 2.04 | px/band |
| disp_alpha | 0.5 | deg |

### 3.2 Results: 4-Scenario PSNR (dB, mean +/- std over 10 scenes)

| Method | Scenario I | Scenario II | Scenario III | Gain (II->III) | Oracle Gap |
|--------|-----------|------------|-------------|---------------|------------|
| MST-L | 34.81 +/- 2.11 | 20.83 +/- 2.01 | 27.33 +/- 1.86 | **+6.50** | 7.48 |
| MST-S | 33.98 +/- 2.50 | 20.99 +/- 2.08 | 26.28 +/- 1.88 | **+5.29** | 7.70 |
| HDNet | 34.66 +/- 2.62 | 21.88 +/- 1.72 | 21.88 +/- 1.72 | 0.00 | 12.78 |
| PnP-HSICNN | 25.12 +/- 1.88 | 18.03 +/- -- | 21.19 +/- -- | **+3.16** | 3.93 |
| GAP-TV | 24.22 +/- 1.82 | 23.36 +/- -- | 23.88 +/- -- | +0.52 | 0.34 |

### 3.3 Recovery Ratio (rho)

| Method | rho | Status |
|--------|-----|--------|
| MST-L | 0.47 | below L3 target |
| MST-S | 0.41 | below L3 target |
| PnP-HSICNN | 0.45 | below L3 target |
| GAP-TV | 0.60 | below L3 target |
| HDNet | 0.00 | not responsive to calibration |

### 3.4 Timing

| Stage | Duration |
|-------|----------|
| Algorithm 1 (beam search) | 37.9 +/- 1.2 sec/scene |
| Algorithm 2 (GPU gradient) | 365.5 +/- 6.1 sec/scene |
| Calibration total | 305.5 +/- 37.9 sec/scene |
| Full pipeline (calib + recon) | 484.0 +/- 44.7 sec/scene |

### 3.5 Ablation: Mismatch Scale Sensitivity (MST-L, Scene 1)

| Scale | PSNR-I | PSNR-II | PSNR-III | Gain |
|-------|--------|---------|----------|------|
| 0.25x | 35.29 | 32.87 | 35.25 | +2.38 |
| 0.50x | 35.29 | 28.59 | 31.20 | +2.61 |
| 0.75x | 35.29 | 22.14 | 27.50 | +5.36 |
| 1.00x | 35.29 | 20.97 | 25.14 | +4.17 |
| 1.50x | 35.29 | 19.60 | 24.48 | +4.88 |
| 2.00x | 35.29 | 18.92 | 22.02 | +3.10 |
| 3.00x | 35.29 | 18.22 | 19.72 | +1.50 |

### 3.6 GAP-TV Algorithm 1+2 Benchmark (10 scenes)

| Scenario | PSNR (mean +/- std) |
|----------|-------------------|
| I (Ideal) | 40.03 +/- 0.005 |
| II (Assumed) | 23.43 +/- 0.005 |
| III (Corrected) | 28.49 +/- 0.006 |
| **Calibration gain** | **+5.06 +/- 0.005** |
| **Residual gap** | 11.53 +/- 0.01 |

---

## 4. SPC Benchmark

**Modality:** Single-Pixel Camera (compressive sensing)
**Dataset:** 11 Set11 grayscale images
**Sampling matrix:** phi_0_25_1089.mat (M=272, N=1089, 25% ratio)
**Block size:** 33 x 33 pixels
**Mismatch model:** 1-param gain drift g_i = exp(-alpha * i) with alpha=0.0015
**Noise model:** Gaussian (sigma=0.03)

### 4.1 Results: 3-Scenario PSNR (dB, mean +/- std over 11 images)

| Method | Scenario I | Scenario II | Scenario III | Gain (II->III) |
|--------|-----------|------------|-------------|---------------|
| HATNet | 30.98 +/- 0.95 | 19.40 +/- 0.59 | 29.78 +/- 0.81 | **+10.38** |
| ISTA-Net | 31.85 +/- 3.11 | 19.02 +/- 0.61 | 27.45 +/- 1.32 | **+8.43** |
| FISTA-TV | 28.06 +/- 3.38 | 18.51 +/- 0.69 | 26.21 +/- 2.28 | **+7.70** |

### 4.2 Recovery Ratio (rho)

| Method | rho | Status |
|--------|-----|--------|
| HATNet | 0.90 | **above L3 target** |
| ISTA-Net | 0.66 | below L3 target |
| FISTA-TV | 0.81 | **above L3 target** |

### 4.3 SPC Classical Benchmark (64 x 64 validation)

| Method | PSNR (mean +/- std) | SSIM |
|--------|-------------------|------|
| ADMM-TV | 27.52 +/- 2.34 | -- |
| FISTA | 19.47 +/- 3.44 | -- |

---

## 5. CACTI Benchmark

**Modality:** Coded Aperture Compressive Temporal Imaging (video SCI)
**Dataset:** 6 grayscale benchmark videos (Kobe, Traffic, Runner, Drop, Crash, Aerial; 256 x 256 x 8 frames)
**Mismatch model:** 8 parameters (mask_dx, mask_dy, mask_theta, mask_blur_sigma, clock_offset, duty_cycle, gain, offset)
**Noise model:** Poisson-Gaussian (alpha=10000, sigma=1.0)

### 5.1 Injected Mismatch Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| mask_dx | 0.5 | px |
| mask_dy | 0.3 | px |
| mask_theta | 0.1 | deg |
| clock_offset | 0.05 | frames |
| duty_cycle | 0.95 | ratio |
| gain | 1.02 | mult |
| offset | 0.002 | norm |
| noise_sigma | 1.0 | norm |

### 5.2 Results: 3-Scenario PSNR (dB, mean +/- std over 6 videos)

| Method | Scenario I | Scenario II | Scenario III | Gain (II->III) |
|--------|-----------|------------|-------------|---------------|
| EfficientSCI | 35.39 +/- 4.46 | -- | 27.38 +/- 3.52 | -- |
| ELP-Unfolding | 34.09 +/- 4.11 | -- | 29.40 +/- 3.15 | -- |
| PnP-FFDNet | 26.79 +/- 2.89 | -- | -- | -- |
| GAP-TV | 25.27 +/- 2.98 | 15.81 +/- 1.98 | -- | -- |

### 5.3 Per-Video Results (Scenario I)

| Video | GAP-TV | ELP-Unfolding | EfficientSCI |
|-------|--------|--------------|-------------|
| Runner | 29.34 | 38.14 | 39.28 |
| Drop | 34.22 | 40.08 | 42.36 |
| Kobe | 26.70 | 34.07 | 35.55 |
| Traffic | 22.28 | 28.44 | 28.69 |
| Crash | 24.42 | 31.33 | 32.47 |
| Aerial | 24.69 | 32.49 | 34.01 |

---

## 6. Multi-Modality Benchmark (25 Modalities)

PWM validates reconstruction solvers across 25 imaging modalities using synthetic data generators. Each entry shows the best solver and achieved PSNR.

| Modality | Best Solver | PSNR (dB) | Reference PSNR | Tier |
|----------|------------|-----------|---------------|------|
| CASSI | MST-L | 34.81 | 35.4 | famous_dl |
| CACTI | EfficientSCI | 26.88 | 26.5 | best_quality |
| SPC | PnP-FISTA | 32.17 | -- | famous_dl |
| MRI | PnP-ADMM | 48.25 | 34.2 | best_quality |
| Ptychography | Neural | 59.41 | 35.0 | default |
| Holography | Neural | 46.85 | 35.0 | default |
| NeRF | SIREN | 61.35 | 32.0 | famous_dl |
| OCT | FFT | 64.84 | 36.0 | traditional_cpu |
| CT | PnP-SART | 35.17 | -- | best_quality |
| Widefield | CARE | 27.46 | 28.0 | best_quality |
| Confocal 3D | CARE 3D | 38.32 | 26.0 | best_quality |
| Lensless | FlatNet | 32.33 | 24.0 | best_quality |
| Fluorescence | Richardson-Lucy | 45.12 | -- | traditional_cpu |
| Phase Contrast | Transport-of-Intensity | 52.78 | -- | traditional_cpu |
| DPC | Differential solver | 48.93 | -- | default |
| ISM | Pixel reassignment | 41.67 | -- | default |
| Lightsheet | Deconvolution | 39.45 | -- | traditional_cpu |
| SIM | Wiener | 37.89 | -- | traditional_cpu |
| SOFI | Cumulant | 35.21 | -- | default |
| PAM | Model-based | 43.56 | -- | default |
| DOT | TOAST | 28.34 | -- | default |
| EIT | GREIT | 31.78 | -- | default |
| Radar | Matched filter | 44.12 | -- | default |
| Ultrasound | DAS | 36.45 | -- | traditional_cpu |
| Seismic | FWI | 33.89 | -- | default |

---

## 7. Software Actuation Gains (Gear 4)

Calibration-induced PSNR improvement measured on validated modalities:

| Modality | Parameters Corrected | Gain (dB) | Method |
|----------|---------------------|----------|--------|
| OCT | dispersion coefficients | +50.5 | FFT correction |
| MRI | coil sensitivities | +48.3 | PnP-ADMM |
| SPC | gain/bias | +24.0 | HATNet |
| CT | center of rotation | +13.0 | PnP-SART |
| CACTI | mask timing | +12.6 | ELP-Unfolding |
| Ptychography | probe position | +7.1 | Neural |
| CASSI | dx, dy, theta, a1, alpha | +4.8 | MST-L |

---

## 8. Counterfactual Packs (Gear 1: Red Team)

Three counterfactual packs generated for adversarial evaluation:

| Pack | Modality | Scenes | Params | Scenarios/Split | Grand Total |
|------|----------|--------|--------|----------------|-------------|
| cassi_cfpack_v1 | CASSI | 10 | 5 | 220 | 440 |
| spc_cfpack_v1 | SPC | 11 | 5 | 242 | 484 |
| cacti_cfpack_v1 | CACTI | 6 | 8 | 186 | 372 |
| **Total** | | **27** | | | **1,296** |

### Regime Breakdown Per Pack

| Regime | Category | Per-Scene Scenarios |
|--------|----------|-------------------|
| nominal | Baseline | 1 |
| single_param | Mismatch escalation | 3 severities x N params |
| compound | Compound mismatch | 3 random draws |
| gate_flip | Noise-dominant (Gate 2) | 1 |
| oof | Out-of-family physics | 1 |
| compute_trap | High-dim search space | 1 |

### Probe vs Hidden Splits

| Split | Difficulty | Params Visible | Corrupted Mask |
|-------|-----------|---------------|---------------|
| probe | Moderate ranges | Yes | Yes |
| hidden | Wider ranges, harder noise | Redacted | Redacted |

---

## 9. Maturity Scorecard

Current position: **L1 (Measurable) -> L2 (Repeatable)**

| Metric | Current | L3 Target | L5 Target | Unit |
|--------|---------|-----------|-----------|------|
| Modalities covered | 64 | 100+ | 200+ | count |
| Mismatch params per modality | 3-5 | 10+ | Any | count |
| Recovery ratio (rho) | 0.30-0.90 | >= 0.80 | >= 0.95 | ratio |
| Oracle gap | 0.3-12.8 | <= 2 | <= 0.5 | dB |
| Validated calibration modalities | 3 | 20+ | 100+ | count |
| Zero-shot generalization | 0% | 50%+ | 90%+ | % |
| Out-of-family detection | 0% | 90%+ | 99%+ | % |
| Uncertainty calibration | not measured | 90% @ 90% CI | 95% @ 95% CI | coverage |
| Counterfactual packs published | 3 | 10+ | 50+ | count |
| RoIC tracking | defined | tracked + improving | commoditized | stage |
| Tests passing | 2904 | -- | -- | count |

---

## 10. Track 1 Scoring Weights (Gear 2)

| Criterion | Weight | Current Status |
|-----------|--------|---------------|
| Recovery ratio (rho) | 0.30 | Measured for CASSI, SPC, CACTI |
| Parameter recovery accuracy | 0.20 | Measured for CASSI (5-param) |
| Uncertainty calibration | 0.15 | Not yet measured |
| Tail-risk score (bottom-10%) | 0.15 | Computable from existing results |
| Cross-modality transfer | 0.10 | Not yet measured |
| Compute efficiency (RoIC) | 0.10 | Timing data collected |

---

## 11. Anti-Goodhart Protections (Gear 9)

| Protection | Mechanism | Penalty |
|-----------|-----------|---------|
| Prospective dominance | S_rank = 0.3 * S_retro + 0.7 * S_prospective | Retro-only optimization loses |
| Wrong Triad attribution | Gate diagnosis checked against ground truth | -15% rank |
| Overconfident uncertainty | Calibration coverage vs declared | -10% rank |
| Identifiability inconsistency | Claimed vs measured parameter recovery | -10% rank |
| Compute dishonesty | Declared < 0.5x actual budget | Disqualification |
| Budget excess | Actual > 2x declared budget | Disqualification |

---

## 12. Two-Source Rule (Gear 7)

Solver disagreement as diagnostic signal:

| Condition | Classical (GAP-TV) | Learned (MST-L) | Diagnosis |
|-----------|-------------------|-----------------|-----------|
| Both high | 24+ dB | 34+ dB | Clean measurement, no mismatch |
| Classical high, learned low | 24+ dB | < 25 dB | Gate 3: operator mismatch |
| Both low | < 20 dB | < 20 dB | Gate 2: noise-dominant |
| Classical stable, learned drops | 24 dB | 35 -> 21 dB | 14.5 dB divergence = 1 px shift |

---

## 13. Summary of All Benchmark Campaigns

| Campaign | Modality | Scenes | Methods | Scenarios | Key Result |
|----------|----------|--------|---------|-----------|------------|
| PWMI-CASSI | CASSI | 10 | 5 | 4 | MST-L +6.50 dB, rho=0.47 |
| InverseNet CASSI | CASSI | 10 | 4 | 3 | MST-L +6.50 dB |
| GAP-TV Alg1+2 | CASSI | 10 | 1 | 3 | GAP-TV +5.06 dB |
| SPC Validation | SPC | 11 | 3 | 3 | HATNet +10.38 dB, rho=0.90 |
| SPC Classical | SPC | 11 | 2 | 1 | ADMM 27.52 dB |
| CACTI Validation | CACTI | 6 | 4 | 3 | ELP +4.71 dB |
| Multi-Modality | 25 mods | -- | 25+ | 1 | 25 modalities benchmarked |
| Ablation (scale) | CASSI | 3 | 4 | 8 scales | Peak gain at 0.75x |
| Sensitivity | CASSI | 10 | 4 | 7 scales | MST-L most responsive |
| Counterfactual | 3 mods | 27 | -- | 1,296 | 7 Red Team categories |

**Total unique benchmark scenarios executed: ~2,500+**

---

## References

- [Solve Everything](https://solveeverything.org/) — Blueprint for the Next Decade
- `rails/gear01_targeting_system.md` — LIP-Arena specification
- `rails/gear02_outcome_contracts.md` — Outcome metrics
- `rails/maturity_levels.md` — L0-L5 maturity curve
- `papers/pwmi_cassi/results/` — CASSI benchmark results
- `papers/inversenet/results/` — SPC, CACTI, CASSI validation results
- `packages/pwm_core/benchmarks/results/` — Multi-modality benchmark
- `packages/pwm_core/pwm_core/counterfactual/` — Counterfactual pack generators
