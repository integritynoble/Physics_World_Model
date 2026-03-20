# Time-of-Flight Depth Camera (`tof_camera`)

**Category**: Depth Imaging | **Canonical DAG**: P --> D | **Carrier**: Photon/IR
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: tv_fista

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Modulation frequency, integration time, multi-path |
| **M1** Synthetic | Prompt tested with synthetic data validation: Modulation frequency, integration time, multi-path |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Modulation frequency, integration time, multi-path |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Modulation frequency, integration time, multi-path |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Time-of-Flight Depth Camera |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: TV-FISTA under multi-path interference, phase wrap |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: TV-FISTA under multi-path interference, phase wrap |
| **M2** Compound | Compound mismatch (3+ params simultaneously): TV-FISTA under multi-path interference, phase wrap |
| **M3** Real Data | Real experimental data with measured mismatch: TV-FISTA under multi-path interference, phase wrap |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: TV-FISTA under multi-path interference, phase wrap |

### Mismatch Parameters
P→D. Multi-path [0,30%], phase wrap +/-1, temperature offset [-5,5] cm.

### Solvers & Expected Performance
- **Solver**: tv_fista

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate multi-path coefficients, wrap count |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate multi-path coefficients, wrap count |
| **M2** Compound | Compound parameter identification (3+ params): Estimate multi-path coefficients, wrap count |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate multi-path coefficients, wrap count |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate multi-path coefficients, wrap count |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct multi-path, phase unwrapping |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct multi-path, phase unwrapping |
| **M2** Compound | Compound correction with rho measurement: Correct multi-path, phase unwrapping |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct multi-path, phase unwrapping |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct multi-path, phase unwrapping |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
