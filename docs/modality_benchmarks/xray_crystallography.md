# X-ray Crystallography (`xray_crystallography`)

**Category**: Scientific Instrumentation | **Canonical DAG**: F --> S --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: direct_methods

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Wavelength, rotation range, detector distance |
| **M1** Synthetic | Prompt tested with synthetic data validation: Wavelength, rotation range, detector distance |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Wavelength, rotation range, detector distance |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Wavelength, rotation range, detector distance |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for X-ray Crystallography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Structure factor extraction under absorption, radiation damage |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Structure factor extraction under absorption, radiation damage |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Structure factor extraction under absorption, radiation damage |
| **M3** Real Data | Real experimental data with measured mismatch: Structure factor extraction under absorption, radiation damage |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Structure factor extraction under absorption, radiation damage |

### Mismatch Parameters
F→S→D. Absorption +/-10%, radiation damage [0,20%], mosaicity +/-50%.

### Solvers & Expected Performance
- **Solver**: direct_methods

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for F --> S --> D: Estimate unit cell, space group, absorption |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate unit cell, space group, absorption |
| **M2** Compound | Compound parameter identification (3+ params): Estimate unit cell, space group, absorption |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate unit cell, space group, absorption |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate unit cell, space group, absorption |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct absorption, scaling, damage |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct absorption, scaling, damage |
| **M2** Compound | Compound correction with rho measurement: Correct absorption, scaling, damage |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct absorption, scaling, damage |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct absorption, scaling, damage |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
