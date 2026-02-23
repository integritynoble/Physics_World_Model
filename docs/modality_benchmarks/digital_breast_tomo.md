# Digital Breast Tomosynthesis (DBT) (`digital_breast_tomo`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: back_projection

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Angular range, projection count, detector type |
| **M1** Synthetic | Prompt tested with synthetic data validation: Angular range, projection count, detector type |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Angular range, projection count, detector type |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Angular range, projection count, detector type |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Digital Breast Tomosynthesis (DBT) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Backprojection under geometric calibration error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Backprojection under geometric calibration error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Backprojection under geometric calibration error |
| **M3** Real Data | Real experimental data with measured mismatch: Backprojection under geometric calibration error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Backprojection under geometric calibration error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Angular range error | 0 | [-2, 2] | deg total |
| Detector motion blur | 0 | [0, 0.5] | px |
| Scatter fraction | 0.3 | [0.1, 0.6] | - |

### Solvers & Expected Performance
- **Solver**: back_projection

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate tube positions, detector flex, compression |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate tube positions, detector flex, compression |
| **M2** Compound | Compound parameter identification (3+ params): Estimate tube positions, detector flex, compression |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate tube positions, detector flex, compression |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate tube positions, detector flex, compression |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct geometry, reduce out-of-plane artifacts |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct geometry, reduce out-of-plane artifacts |
| **M2** Compound | Compound correction with rho measurement: Correct geometry, reduce out-of-plane artifacts |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct geometry, reduce out-of-plane artifacts |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct geometry, reduce out-of-plane artifacts |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
