# X-ray NDT (Radiography) (`xray_ndt`)

**Category**: Industrial Inspection | **Canonical DAG**: Pi --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: contrast_enhancement

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Source type, film/DR detector, exposure chart |
| **M1** Synthetic | Prompt tested with synthetic data validation: Source type, film/DR detector, exposure chart |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Source type, film/DR detector, exposure chart |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Source type, film/DR detector, exposure chart |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for X-ray NDT (Radiography) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Enhancement under scatter, geometric unsharpness |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Enhancement under scatter, geometric unsharpness |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Enhancement under scatter, geometric unsharpness |
| **M3** Real Data | Real experimental data with measured mismatch: Enhancement under scatter, geometric unsharpness |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Enhancement under scatter, geometric unsharpness |

### Mismatch Parameters
Pi→D. SDD +/-5%, geometric unsharpness [0,1] mm.

### Solvers & Expected Performance
- **Solver**: contrast_enhancement

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate SDD, source size, scatter buildup |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate SDD, source size, scatter buildup |
| **M2** Compound | Compound parameter identification (3+ params): Estimate SDD, source size, scatter buildup |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate SDD, source size, scatter buildup |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate SDD, source size, scatter buildup |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct scatter, magnification, contrast |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct scatter, magnification, contrast |
| **M2** Compound | Compound correction with rho measurement: Correct scatter, magnification, contrast |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct scatter, magnification, contrast |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct scatter, magnification, contrast |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
