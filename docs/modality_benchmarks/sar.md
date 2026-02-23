# Synthetic Aperture Radar (SAR) (`sar`)

**Category**: Remote Sensing | **Canonical DAG**: F --> D | **Carrier**: RF
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: backprojection

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Bandwidth, PRF, look angle, aperture length |
| **M1** Synthetic | Prompt tested with synthetic data validation: Bandwidth, PRF, look angle, aperture length |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Bandwidth, PRF, look angle, aperture length |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Bandwidth, PRF, look angle, aperture length |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Synthetic Aperture Radar (SAR) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Backprojection under motion error, autofocus |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Backprojection under motion error, autofocus |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Backprojection under motion error, autofocus |
| **M3** Real Data | Real experimental data with measured mismatch: Backprojection under motion error, autofocus |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Backprojection under motion error, autofocus |

### Mismatch Parameters
F→D, RF. Velocity +/-1%, motion phase [0,pi/4] rad.

### Solvers & Expected Performance
- **Solver**: backprojection

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for F --> D: Estimate platform motion errors, phase history |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate platform motion errors, phase history |
| **M2** Compound | Compound parameter identification (3+ params): Estimate platform motion errors, phase history |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate platform motion errors, phase history |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate platform motion errors, phase history |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct autofocus, motion compensation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct autofocus, motion compensation |
| **M2** Compound | Compound correction with rho measurement: Correct autofocus, motion compensation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct autofocus, motion compensation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct autofocus, motion compensation |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
