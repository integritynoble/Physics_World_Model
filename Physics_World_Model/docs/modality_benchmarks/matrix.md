# Generic Matrix Sensing (`matrix`)

**Category**: Compressive Imaging | **Canonical DAG**: M --> D | **Carrier**: Photon
**Current Maturity**: M1 | **Forward Model**: explicit_matrix | **Default Solver**: fista_l2

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Measurement matrix design (RIP, coherence), conditioning |
| **M1** Synthetic | Prompt tested with synthetic data validation: Measurement matrix design (RIP, coherence), conditioning |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Measurement matrix design (RIP, coherence), conditioning |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Measurement matrix design (RIP, coherence), conditioning |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Generic Matrix Sensing |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: CG/ADMM under matrix perturbation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: CG/ADMM under matrix perturbation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): CG/ADMM under matrix perturbation |
| **M3** Real Data | Real experimental data with measured mismatch: CG/ADMM under matrix perturbation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: CG/ADMM under matrix perturbation |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | True Example | Unit |
|-----------|---------|----------------|--------------|------|
| Matrix perturbation | 0 | [0, 10%] of |  | A |
| Condition number change | kappa | [kappa, 10*kappa] | - |  |

### Solvers & Expected Performance
- **Solver**: fista_l2

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> D: Estimate matrix condition number, perturbation magnitude |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate matrix condition number, perturbation magnitude |
| **M2** Compound | Compound parameter identification (3+ params): Estimate matrix condition number, perturbation magnitude |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate matrix condition number, perturbation magnitude |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate matrix condition number, perturbation magnitude |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct matrix calibration errors |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct matrix calibration errors |
| **M2** Compound | Compound correction with rho measurement: Correct matrix calibration errors |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct matrix calibration errors |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct matrix calibration errors |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
