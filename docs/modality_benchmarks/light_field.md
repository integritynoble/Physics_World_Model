# Light Field Imaging (`light_field`)

**Category**: Computational Optics | **Canonical DAG**: C --> S --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: shift_and_sum

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Microlens array, angular vs spatial resolution |
| **M1** Synthetic | Prompt tested with synthetic data validation: Microlens array, angular vs spatial resolution |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Microlens array, angular vs spatial resolution |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Microlens array, angular vs spatial resolution |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Light Field Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Shift-and-sum under microlens alignment error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Shift-and-sum under microlens alignment error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Shift-and-sum under microlens alignment error |
| **M3** Real Data | Real experimental data with measured mismatch: Shift-and-sum under microlens alignment error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Shift-and-sum under microlens alignment error |

### Mismatch Parameters
C→S→D. Microlens pitch +/-2%, rotation [-1,1] deg, f-number +/-30%.

### Solvers & Expected Performance
- **Solver**: shift_and_sum

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> S --> D: Estimate microlens pitch, rotation, f-number |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate microlens pitch, rotation, f-number |
| **M2** Compound | Compound parameter identification (3+ params): Estimate microlens pitch, rotation, f-number |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate microlens pitch, rotation, f-number |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate microlens pitch, rotation, f-number |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct microlens calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct microlens calibration |
| **M2** Compound | Compound correction with rho measurement: Correct microlens calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct microlens calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct microlens calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
