# Optical Coherence Tomography (OCT) (`oct`)

**Category**: Medical Imaging | **Canonical DAG**: P+P --> Sigma --> D | **Carrier**: Photon
**Current Maturity**: M1 | **Forward Model**: nonlinear_operator | **Default Solver**: fft_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Source bandwidth, reference arm, scan pattern, axial resolution |
| **M1** Synthetic | Prompt tested with synthetic data validation: Source bandwidth, reference arm, scan pattern, axial resolution |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Source bandwidth, reference arm, scan pattern, axial resolution |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Source bandwidth, reference arm, scan pattern, axial resolution |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Optical Coherence Tomography (OCT) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: FFT recon under dispersion mismatch, reference arm drift |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: FFT recon under dispersion mismatch, reference arm drift |
| **M2** Compound | Compound mismatch (3+ params simultaneously): FFT recon under dispersion mismatch, reference arm drift |
| **M3** Real Data | Real experimental data with measured mismatch: FFT recon under dispersion mismatch, reference arm drift |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: FFT recon under dispersion mismatch, reference arm drift |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Dispersion GDD | 0 | [-100, 100] | fs^2 |
| Reference arm position | optimal | +/- 50 | um |
| K-linearization error | 0 | [0, 0.5%] | relative |

### Solvers & Expected Performance
- **Solver**: fft_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P+P --> Sigma --> D: Estimate dispersion coefficients, reference arm position |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate dispersion coefficients, reference arm position |
| **M2** Compound | Compound parameter identification (3+ params): Estimate dispersion coefficients, reference arm position |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate dispersion coefficients, reference arm position |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate dispersion coefficients, reference arm position |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct dispersion, reference drift |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct dispersion, reference drift |
| **M2** Compound | Compound correction with rho measurement: Correct dispersion, reference drift |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct dispersion, reference drift |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct dispersion, reference drift |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
