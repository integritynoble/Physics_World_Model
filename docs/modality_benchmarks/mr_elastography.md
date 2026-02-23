# MR Elastography (MRE) (`mr_elastography`)

**Category**: Medical Imaging | **Canonical DAG**: M --> F --> S --> D | **Carrier**: Spin/RF
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: lfe_inversion

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Driver frequency, MEG encoding, inversion algorithm |
| **M1** Synthetic | Prompt tested with synthetic data validation: Driver frequency, MEG encoding, inversion algorithm |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Driver frequency, MEG encoding, inversion algorithm |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Driver frequency, MEG encoding, inversion algorithm |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for MR Elastography (MRE) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: LFE/DI under wave attenuation, reflection, boundary |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: LFE/DI under wave attenuation, reflection, boundary |
| **M2** Compound | Compound mismatch (3+ params simultaneously): LFE/DI under wave attenuation, reflection, boundary |
| **M3** Real Data | Real experimental data with measured mismatch: LFE/DI under wave attenuation, reflection, boundary |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: LFE/DI under wave attenuation, reflection, boundary |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Shear wave frequency error | 0 | [-10%, 10%] | - |
| Wave attenuation model | correct | +/- 20% | - |
| Motion encoding gradient error | 0 | [-5%, 5%] | - |
| Boundary reflection | none | [0, 20%] | amplitude |

### Solvers & Expected Performance
- **Solver**: lfe_inversion

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> F --> S --> D: Estimate shear modulus, attenuation, wave speed |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate shear modulus, attenuation, wave speed |
| **M2** Compound | Compound parameter identification (3+ params): Estimate shear modulus, attenuation, wave speed |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate shear modulus, attenuation, wave speed |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate shear modulus, attenuation, wave speed |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct wave model, boundary conditions |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct wave model, boundary conditions |
| **M2** Compound | Compound correction with rho measurement: Correct wave model, boundary conditions |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct wave model, boundary conditions |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct wave model, boundary conditions |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
