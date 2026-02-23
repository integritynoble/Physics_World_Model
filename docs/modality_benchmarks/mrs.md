# MR Spectroscopy (MRS) (`mrs`)

**Category**: Medical Imaging | **Canonical DAG**: M --> F --> S --> D | **Carrier**: Spin/RF
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: lcmodel

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Voxel localization, spectral bandwidth, water suppression |
| **M1** Synthetic | Prompt tested with synthetic data validation: Voxel localization, spectral bandwidth, water suppression |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Voxel localization, spectral bandwidth, water suppression |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Voxel localization, spectral bandwidth, water suppression |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for MR Spectroscopy (MRS) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: LCModel fitting under lineshape distortion, baseline error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: LCModel fitting under lineshape distortion, baseline error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): LCModel fitting under lineshape distortion, baseline error |
| **M3** Real Data | Real experimental data with measured mismatch: LCModel fitting under lineshape distortion, baseline error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: LCModel fitting under lineshape distortion, baseline error |

### Mismatch Parameters
M→F→S→D, Spin/RF. Lineshape, eddy phase [0,0.5] rad, residual water [0,100x].

### Solvers & Expected Performance
- **Solver**: lcmodel

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> F --> S --> D: Estimate lineshape, eddy current phase, residual water |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate lineshape, eddy current phase, residual water |
| **M2** Compound | Compound parameter identification (3+ params): Estimate lineshape, eddy current phase, residual water |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate lineshape, eddy current phase, residual water |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate lineshape, eddy current phase, residual water |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct lineshape, eddy current, baseline |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct lineshape, eddy current, baseline |
| **M2** Compound | Compound correction with rho measurement: Correct lineshape, eddy current, baseline |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct lineshape, eddy current, baseline |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct lineshape, eddy current, baseline |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
