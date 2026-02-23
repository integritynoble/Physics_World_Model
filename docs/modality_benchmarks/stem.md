# Scanning Transmission Electron Microscopy (STEM) (`stem`)

**Category**: Electron Microscopy | **Canonical DAG**: S --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: direct_imaging

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Convergence angle, detector geometry, scan pattern |
| **M1** Synthetic | Prompt tested with synthetic data validation: Convergence angle, detector geometry, scan pattern |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Convergence angle, detector geometry, scan pattern |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Convergence angle, detector geometry, scan pattern |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Scanning Transmission Electron Microscopy (STEM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Direct imaging under scan distortion, probe aberration |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Direct imaging under scan distortion, probe aberration |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Direct imaging under scan distortion, probe aberration |
| **M3** Real Data | Real experimental data with measured mismatch: Direct imaging under scan distortion, probe aberration |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Direct imaging under scan distortion, probe aberration |

### Mismatch Parameters
S→D, Electron. Scan distortion [0,3] px, probe aberration [-50,50] nm.

### Solvers & Expected Performance
- **Solver**: direct_imaging

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for S --> D: Estimate scan distortion, probe parameters |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate scan distortion, probe parameters |
| **M2** Compound | Compound parameter identification (3+ params): Estimate scan distortion, probe parameters |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate scan distortion, probe parameters |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate scan distortion, probe parameters |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct scan calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct scan calibration |
| **M2** Compound | Compound correction with rho measurement: Correct scan calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct scan calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct scan calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
