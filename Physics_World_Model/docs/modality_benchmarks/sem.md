# Scanning Electron Microscopy (SEM) (`sem`)

**Category**: Electron Microscopy | **Canonical DAG**: C --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: direct_imaging

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Beam energy, working distance, detector type |
| **M1** Synthetic | Prompt tested with synthetic data validation: Beam energy, working distance, detector type |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Beam energy, working distance, detector type |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Beam energy, working distance, detector type |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Scanning Electron Microscopy (SEM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Direct imaging under charging, drift, astigmatism |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Direct imaging under charging, drift, astigmatism |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Direct imaging under charging, drift, astigmatism |
| **M3** Real Data | Real experimental data with measured mismatch: Direct imaging under charging, drift, astigmatism |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Direct imaging under charging, drift, astigmatism |

### Mismatch Parameters
C→D, Electron. Astigmatism [0,50] nm, drift [0,1] nm/s, charging [0,500] V.

### Solvers & Expected Performance
- **Solver**: direct_imaging

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate stigmation, working distance, drift rate |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate stigmation, working distance, drift rate |
| **M2** Compound | Compound parameter identification (3+ params): Estimate stigmation, working distance, drift rate |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate stigmation, working distance, drift rate |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate stigmation, working distance, drift rate |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct astigmatism, drift compensation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct astigmatism, drift compensation |
| **M2** Compound | Compound correction with rho measurement: Correct astigmatism, drift compensation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct astigmatism, drift compensation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct astigmatism, drift compensation |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
