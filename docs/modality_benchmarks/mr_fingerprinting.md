# MR Fingerprinting (MRF) (`mr_fingerprinting`)

**Category**: Medical Imaging | **Canonical DAG**: M --> F --> S --> D | **Carrier**: Spin/RF
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: dictionary_matching

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Flip angle pattern, TR pattern, dictionary design |
| **M1** Synthetic | Prompt tested with synthetic data validation: Flip angle pattern, TR pattern, dictionary design |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Flip angle pattern, TR pattern, dictionary design |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Flip angle pattern, TR pattern, dictionary design |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for MR Fingerprinting (MRF) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Dictionary matching under B0/B1 variation, aliasing |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Dictionary matching under B0/B1 variation, aliasing |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Dictionary matching under B0/B1 variation, aliasing |
| **M3** Real Data | Real experimental data with measured mismatch: Dictionary matching under B0/B1 variation, aliasing |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Dictionary matching under B0/B1 variation, aliasing |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Dictionary resolution (T1, T2) | fine | coarse [2x, 5x] | - |
| B1 inhomogeneity | 0 | [0, 15%] | - |
| Undersampling artifact | 0 | [0, 20%] | - |

### Solvers & Expected Performance
- **Solver**: dictionary_matching

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> F --> S --> D: Estimate T1, T2, B0, B1 maps simultaneously |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate T1, T2, B0, B1 maps simultaneously |
| **M2** Compound | Compound parameter identification (3+ params): Estimate T1, T2, B0, B1 maps simultaneously |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate T1, T2, B0, B1 maps simultaneously |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate T1, T2, B0, B1 maps simultaneously |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct B0/B1, refine dictionary matching |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct B0/B1, refine dictionary matching |
| **M2** Compound | Compound correction with rho measurement: Correct B0/B1, refine dictionary matching |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct B0/B1, refine dictionary matching |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct B0/B1, refine dictionary matching |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
