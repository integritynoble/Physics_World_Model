# Single-Pixel Camera (SPC) (`spc`)

**Category**: Compressive Imaging | **Canonical DAG**: M --> Sigma --> D | **Carrier**: Photon
**Current Maturity**: M3 | **Forward Model**: explicit_matrix | **Default Solver**: pnp_fista

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Pattern type (Hadamard/Gaussian), sampling rate, DMD resolution |
| **M1** Synthetic | Prompt tested with synthetic data validation: Pattern type (Hadamard/Gaussian), sampling rate, DMD resolution |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Pattern type (Hadamard/Gaussian), sampling rate, DMD resolution |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Pattern type (Hadamard/Gaussian), sampling rate, DMD resolution |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Single-Pixel Camera (SPC) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: FISTA-TV under gain drift (alpha) and measurement noise (sigma_y) |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: FISTA-TV under gain drift (alpha) and measurement noise (sigma_y) |
| **M2** Compound | Compound mismatch (3+ params simultaneously): FISTA-TV under gain drift (alpha) and measurement noise (sigma_y) |
| **M3** Real Data | Real experimental data with measured mismatch: FISTA-TV under gain drift (alpha) and measurement noise (sigma_y) |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: FISTA-TV under gain drift (alpha) and measurement noise (sigma_y) |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Gain drift alpha | 1.0 | [0.8, 1.2] | - |
| Measurement noise sigma_y | 0.01 | [0, 0.1] | - |
| Pattern error (bit flips) | 0 | [0, 1%] | - |

### Solvers & Expected Performance
- **Solver**: pnp_fista
- **Validated baseline**: FISTA-TV +7.71 dB, rho = 0.86

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> Sigma --> D: Estimate gain drift curve and noise level |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate gain drift curve and noise level |
| **M2** Compound | Compound parameter identification (3+ params): Estimate gain drift curve and noise level |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate gain drift curve and noise level |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate gain drift curve and noise level |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct gain model; rho validated at 86% |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct gain model; rho validated at 86% |
| **M2** Compound | Compound correction with rho measurement: Correct gain model; rho validated at 86% |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct gain model; rho validated at 86% |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct gain model; rho validated at 86% |

### Correction Targets
- **Expected rho**: 0.86

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
