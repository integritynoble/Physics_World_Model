# Functional MRI (BOLD fMRI) (`fmri`)

**Category**: Medical Imaging | **Canonical DAG**: M --> F --> S --> D | **Carrier**: Spin/RF
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: sense

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: TR/TE, spatial resolution, temporal resolution, EPI trajectory |
| **M1** Synthetic | Prompt tested with synthetic data validation: TR/TE, spatial resolution, temporal resolution, EPI trajectory |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for TR/TE, spatial resolution, temporal resolution, EPI trajectory |
| **M3** Real Data | Grounded in real experimental/clinical protocols: TR/TE, spatial resolution, temporal resolution, EPI trajectory |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Functional MRI (BOLD fMRI) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: SENSE + GLM under geometric distortion, signal dropout |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: SENSE + GLM under geometric distortion, signal dropout |
| **M2** Compound | Compound mismatch (3+ params simultaneously): SENSE + GLM under geometric distortion, signal dropout |
| **M3** Real Data | Real experimental data with measured mismatch: SENSE + GLM under geometric distortion, signal dropout |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: SENSE + GLM under geometric distortion, signal dropout |

### Mismatch Parameters
M→F→S→D, Spin/RF. Distortion [0,5] px, dropout [0,15%], motion 6DOF [0,3] mm/deg.

### Solvers & Expected Performance
- **Solver**: sense

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> F --> S --> D: Estimate field map, distortion, motion parameters |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate field map, distortion, motion parameters |
| **M2** Compound | Compound parameter identification (3+ params): Estimate field map, distortion, motion parameters |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate field map, distortion, motion parameters |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate field map, distortion, motion parameters |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct distortion, motion, physiological noise |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct distortion, motion, physiological noise |
| **M2** Compound | Compound correction with rho measurement: Correct distortion, motion, physiological noise |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct distortion, motion, physiological noise |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct distortion, motion, physiological noise |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
