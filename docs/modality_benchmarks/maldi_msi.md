# MALDI Mass Spectrometry Imaging (`maldi_msi`)

**Category**: Scientific Instrumentation | **Canonical DAG**: S --> D | **Carrier**: Ion
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: peak_picking

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Laser spot size, step size, matrix application |
| **M1** Synthetic | Prompt tested with synthetic data validation: Laser spot size, step size, matrix application |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Laser spot size, step size, matrix application |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Laser spot size, step size, matrix application |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for MALDI Mass Spectrometry Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Ion image under matrix inhomogeneity, mass drift |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Ion image under matrix inhomogeneity, mass drift |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Ion image under matrix inhomogeneity, mass drift |
| **M3** Real Data | Real experimental data with measured mismatch: Ion image under matrix inhomogeneity, mass drift |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Ion image under matrix inhomogeneity, mass drift |

### Mismatch Parameters
S→D. Mass drift [-10,10] ppm, matrix inhomogeneity [0,30%].

### Solvers & Expected Performance
- **Solver**: peak_picking

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for S --> D: Estimate mass calibration drift, ion suppression |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate mass calibration drift, ion suppression |
| **M2** Compound | Compound parameter identification (3+ params): Estimate mass calibration drift, ion suppression |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate mass calibration drift, ion suppression |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate mass calibration drift, ion suppression |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct mass calibration, normalize |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct mass calibration, normalize |
| **M2** Compound | Compound correction with rho measurement: Correct mass calibration, normalize |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct mass calibration, normalize |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct mass calibration, normalize |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
