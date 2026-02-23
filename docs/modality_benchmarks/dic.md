# Differential Interference Contrast (DIC) (`dic`)

**Category**: Microscopy | **Canonical DAG**: M --> C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: dic_gradient_integration

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Prism shear, bias retardation, NA selection |
| **M1** Synthetic | Prompt tested with synthetic data validation: Prism shear, bias retardation, NA selection |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Prism shear, bias retardation, NA selection |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Prism shear, bias retardation, NA selection |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Differential Interference Contrast (DIC) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: DIC gradient reconstruction under bias error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: DIC gradient reconstruction under bias error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): DIC gradient reconstruction under bias error |
| **M3** Real Data | Real experimental data with measured mismatch: DIC gradient reconstruction under bias error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: DIC gradient reconstruction under bias error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Shear amount | 100 | [50, 200] | nm |
| Bias retardation | lambda/4 | +/- 30 nm | nm |
| Prism orientation | 0 | [-3, 3] | deg |

### Solvers & Expected Performance
- **Solver**: dic_gradient_integration

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> C --> D: Estimate shear distance, bias offset, extinction ratio |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate shear distance, bias offset, extinction ratio |
| **M2** Compound | Compound parameter identification (3+ params): Estimate shear distance, bias offset, extinction ratio |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate shear distance, bias offset, extinction ratio |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate shear distance, bias offset, extinction ratio |

### True-Spec Parameters
Shear distance, bias, prism angle

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct bias retardation, prism alignment |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct bias retardation, prism alignment |
| **M2** Compound | Compound correction with rho measurement: Correct bias retardation, prism alignment |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct bias retardation, prism alignment |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct bias retardation, prism alignment |

### Correction Targets
- **Expected rho**: >= 0.70

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
