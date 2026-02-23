# Focused Ion Beam SEM (FIB-SEM) (`fib_sem`)

**Category**: Electron Microscopy | **Canonical DAG**: S --> C --> D | **Carrier**: Electron + Ion
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: stack_alignment

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Milling current, slice thickness, imaging kV |
| **M1** Synthetic | Prompt tested with synthetic data validation: Milling current, slice thickness, imaging kV |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Milling current, slice thickness, imaging kV |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Milling current, slice thickness, imaging kV |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Focused Ion Beam SEM (FIB-SEM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: 3D stack under curtaining, charging, slice thickness variation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: 3D stack under curtaining, charging, slice thickness variation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): 3D stack under curtaining, charging, slice thickness variation |
| **M3** Real Data | Real experimental data with measured mismatch: 3D stack under curtaining, charging, slice thickness variation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: 3D stack under curtaining, charging, slice thickness variation |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Slice thickness variation | 0 | [0, 15%] | - |
| Curtaining artifact | 0 | [0, 0.3] | relative |
| Charging | 0 | [0, 300] | V |
| Drift between slices | 0 | [0, 5] | nm |

### Solvers & Expected Performance
- **Solver**: stack_alignment

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for S --> C --> D: Estimate slice thickness, curtain artifact, charging |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate slice thickness, curtain artifact, charging |
| **M2** Compound | Compound parameter identification (3+ params): Estimate slice thickness, curtain artifact, charging |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate slice thickness, curtain artifact, charging |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate slice thickness, curtain artifact, charging |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct curtaining, align slices, normalize intensity |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct curtaining, align slices, normalize intensity |
| **M2** Compound | Compound correction with rho measurement: Correct curtaining, align slices, normalize intensity |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct curtaining, align slices, normalize intensity |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct curtaining, align slices, normalize intensity |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
