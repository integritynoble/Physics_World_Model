# Magnetic Particle Imaging (MPI) (`magnetic_particle`)

**Category**: Broader Experimental Science | **Canonical DAG**: M --> F --> D | **Carrier**: Magnetic
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: system_function_inversion

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: FFP trajectory, drive field, selection field |
| **M1** Synthetic | Prompt tested with synthetic data validation: FFP trajectory, drive field, selection field |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for FFP trajectory, drive field, selection field |
| **M3** Real Data | Grounded in real experimental/clinical protocols: FFP trajectory, drive field, selection field |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Magnetic Particle Imaging (MPI) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: System function inversion under relaxation effects |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: System function inversion under relaxation effects |
| **M2** Compound | Compound mismatch (3+ params simultaneously): System function inversion under relaxation effects |
| **M3** Real Data | Real experimental data with measured mismatch: System function inversion under relaxation effects |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: System function inversion under relaxation effects |

### Mismatch Parameters
M→F→D. System function [0,10%], relaxation [0,20%].

### Solvers & Expected Performance
- **Solver**: system_function_inversion

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> F --> D: Estimate system function, relaxation params |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate system function, relaxation params |
| **M2** Compound | Compound parameter identification (3+ params): Estimate system function, relaxation params |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate system function, relaxation params |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate system function, relaxation params |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct system function, resolution |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct system function, resolution |
| **M2** Compound | Compound correction with rho measurement: Correct system function, resolution |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct system function, resolution |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct system function, resolution |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
