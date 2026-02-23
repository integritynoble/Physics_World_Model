# Electron Tomography (`electron_tomography`)

**Category**: Electron Microscopy | **Canonical DAG**: Pi --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: sirt

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Tilt range, tilt increment, missing wedge |
| **M1** Synthetic | Prompt tested with synthetic data validation: Tilt range, tilt increment, missing wedge |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Tilt range, tilt increment, missing wedge |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Tilt range, tilt increment, missing wedge |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Electron Tomography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: SIRT/WBP under tilt axis misalignment, magnification change |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: SIRT/WBP under tilt axis misalignment, magnification change |
| **M2** Compound | Compound mismatch (3+ params simultaneously): SIRT/WBP under tilt axis misalignment, magnification change |
| **M3** Real Data | Real experimental data with measured mismatch: SIRT/WBP under tilt axis misalignment, magnification change |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: SIRT/WBP under tilt axis misalignment, magnification change |

### Mismatch Parameters
Pi→D, Electron. Tilt axis [-3,3] px, mag variation +/-2%, missing wedge [20,50] deg.

### Solvers & Expected Performance
- **Solver**: sirt

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate tilt axis offset, magnification variation |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate tilt axis offset, magnification variation |
| **M2** Compound | Compound parameter identification (3+ params): Estimate tilt axis offset, magnification variation |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate tilt axis offset, magnification variation |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate tilt axis offset, magnification variation |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct tilt axis, missing wedge |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct tilt axis, missing wedge |
| **M2** Compound | Compound correction with rho measurement: Correct tilt axis, missing wedge |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct tilt axis, missing wedge |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct tilt axis, missing wedge |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
