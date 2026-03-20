# Electron Backscatter Diffraction (EBSD) (`ebsd`)

**Category**: Electron Microscopy | **Canonical DAG**: R --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: hough_indexing

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Tilt angle, step size, detector geometry |
| **M1** Synthetic | Prompt tested with synthetic data validation: Tilt angle, step size, detector geometry |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Tilt angle, step size, detector geometry |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Tilt angle, step size, detector geometry |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Electron Backscatter Diffraction (EBSD) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Hough indexing under pattern center error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Hough indexing under pattern center error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Hough indexing under pattern center error |
| **M3** Real Data | Real experimental data with measured mismatch: Hough indexing under pattern center error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Hough indexing under pattern center error |

### Mismatch Parameters
R→D, Electron. Pattern center +/-2%, detector tilt [-1,1] deg.

### Solvers & Expected Performance
- **Solver**: hough_indexing

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for R --> D: Estimate pattern center (PC), detector tilt |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate pattern center (PC), detector tilt |
| **M2** Compound | Compound parameter identification (3+ params): Estimate pattern center (PC), detector tilt |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate pattern center (PC), detector tilt |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate pattern center (PC), detector tilt |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct PC calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct PC calibration |
| **M2** Compound | Compound correction with rho measurement: Correct PC calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct PC calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct PC calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
