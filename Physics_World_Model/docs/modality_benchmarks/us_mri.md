# US/MRI Fusion (`us_mri`)

**Category**: Multi-Modal Fusion | **Canonical DAG**: P --> D (US) + M --> F --> S --> D (MR) --> Fusion | **Carrier**: Acoustic + RF
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: registration_fusion

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Registration method, US probe tracking, MR sequence |
| **M1** Synthetic | Prompt tested with synthetic data validation: Registration method, US probe tracking, MR sequence |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Registration method, US probe tracking, MR sequence |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Registration method, US probe tracking, MR sequence |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for US/MRI Fusion |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Fused guidance under probe drift, tissue deformation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Fused guidance under probe drift, tissue deformation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Fused guidance under probe drift, tissue deformation |
| **M3** Real Data | Real experimental data with measured mismatch: Fused guidance under probe drift, tissue deformation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Fused guidance under probe drift, tissue deformation |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Registration error (deformable) | 0 | [0, 10] | mm |
| Probe pressure deformation | 0 | [0, 15] | mm |
| MR distortion | 0 | [0, 5] | mm |

### Solvers & Expected Performance
- **Solver**: registration_fusion

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D (US) + M --> F --> S --> D (MR) --> Fusion: Estimate probe position drift, tissue deformation field |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate probe position drift, tissue deformation field |
| **M2** Compound | Compound parameter identification (3+ params): Estimate probe position drift, tissue deformation field |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate probe position drift, tissue deformation field |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate probe position drift, tissue deformation field |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct registration, deformation compensation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct registration, deformation compensation |
| **M2** Compound | Compound correction with rho measurement: Correct registration, deformation compensation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct registration, deformation compensation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct registration, deformation compensation |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
