# X-ray Fluorescence Tomography (`xrf_tomo`)

**Category**: Scientific Instrumentation | **Canonical DAG**: Pi --> R --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: fbp_self_absorption

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Excitation energy, rotation steps, self-absorption |
| **M1** Synthetic | Prompt tested with synthetic data validation: Excitation energy, rotation steps, self-absorption |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Excitation energy, rotation steps, self-absorption |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Excitation energy, rotation steps, self-absorption |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for X-ray Fluorescence Tomography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Filtered backprojection under self-absorption error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Filtered backprojection under self-absorption error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Filtered backprojection under self-absorption error |
| **M3** Real Data | Real experimental data with measured mismatch: Filtered backprojection under self-absorption error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Filtered backprojection under self-absorption error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Self-absorption correction | 0 | [0, 30%] | - |
| Rotation axis offset | 0 | [-3, 3] | px |
| Fluorescence yield error | 0 | [-10%, 10%] | - |
| Dead time at high count rate | 0 | [0, 10%] | - |

### Solvers & Expected Performance
- **Solver**: fbp_self_absorption

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> R --> D: Estimate attenuation map, fluorescence yield |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate attenuation map, fluorescence yield |
| **M2** Compound | Compound parameter identification (3+ params): Estimate attenuation map, fluorescence yield |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate attenuation map, fluorescence yield |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate attenuation map, fluorescence yield |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct self-absorption, attenuation compensation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct self-absorption, attenuation compensation |
| **M2** Compound | Compound correction with rho measurement: Correct self-absorption, attenuation compensation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct self-absorption, attenuation compensation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct self-absorption, attenuation compensation |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
