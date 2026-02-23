# Cryo-Electron Tomography (Cryo-ET) (`cryo_et`)

**Category**: Electron Microscopy | **Canonical DAG**: Pi --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: sirt

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Tilt scheme, dose fractionation, defocus |
| **M1** Synthetic | Prompt tested with synthetic data validation: Tilt scheme, dose fractionation, defocus |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Tilt scheme, dose fractionation, defocus |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Tilt scheme, dose fractionation, defocus |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Cryo-Electron Tomography (Cryo-ET) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: SIRT under beam-induced motion, dose damage |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: SIRT under beam-induced motion, dose damage |
| **M2** Compound | Compound mismatch (3+ params simultaneously): SIRT under beam-induced motion, dose damage |
| **M3** Real Data | Real experimental data with measured mismatch: SIRT under beam-induced motion, dose damage |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: SIRT under beam-induced motion, dose damage |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Tilt axis offset | 0 | [-3, 3] | px |
| Tilt angle accuracy | 0 | [-1, 1] | deg per tilt |
| Dose-induced shrinkage | 0 | [0, 10%] | - |
| CTF per-tilt variation | varies | +/- 0.5 um defocus | um |
| Missing wedge | 30 | [20, 50] | deg |

### Solvers & Expected Performance
- **Solver**: sirt

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate per-tilt motion, CTF, accumulated dose |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate per-tilt motion, CTF, accumulated dose |
| **M2** Compound | Compound parameter identification (3+ params): Estimate per-tilt motion, CTF, accumulated dose |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate per-tilt motion, CTF, accumulated dose |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate per-tilt motion, CTF, accumulated dose |

### True-Spec Parameters
Tilt axis, angles, defocus per tilt, shrinkage trajectory, ice thickness

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct motion, CTF per tilt, dose weighting |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct motion, CTF per tilt, dose weighting |
| **M2** Compound | Compound correction with rho measurement: Correct motion, CTF per tilt, dose weighting |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct motion, CTF per tilt, dose weighting |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct motion, CTF per tilt, dose weighting |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Subtomogram averaging benchmark, SIRT vs WBP comparison.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
