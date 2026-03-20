# Shearography (`shearography`)

**Category**: Industrial Inspection | **Canonical DAG**: M --> P --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: phase_unwrap

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Shearing amount, illumination, loading method |
| **M1** Synthetic | Prompt tested with synthetic data validation: Shearing amount, illumination, loading method |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Shearing amount, illumination, loading method |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Shearing amount, illumination, loading method |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Shearography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Phase map under decorrelation, rigid body motion |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Phase map under decorrelation, rigid body motion |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Phase map under decorrelation, rigid body motion |
| **M3** Real Data | Real experimental data with measured mismatch: Phase map under decorrelation, rigid body motion |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Phase map under decorrelation, rigid body motion |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Shearing amount error | 0 | [-10%, 10%] | - |
| Speckle decorrelation | 0 | [0, 0.3] | - |
| Loading non-uniformity | 0 | [0, 20%] | - |

### Solvers & Expected Performance
- **Solver**: phase_unwrap

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> P --> D: Estimate shear distance, decorrelation rate |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate shear distance, decorrelation rate |
| **M2** Compound | Compound parameter identification (3+ params): Estimate shear distance, decorrelation rate |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate shear distance, decorrelation rate |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate shear distance, decorrelation rate |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct rigid body motion, phase unwrapping |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct rigid body motion, phase unwrapping |
| **M2** Compound | Compound correction with rho measurement: Correct rigid body motion, phase unwrapping |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct rigid body motion, phase unwrapping |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct rigid body motion, phase unwrapping |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
