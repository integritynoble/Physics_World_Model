# Structured-Light Depth Camera (`structured_light`)

**Category**: Depth Imaging | **Canonical DAG**: M --> C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: phase_unwrap

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Pattern type, projector-camera geometry |
| **M1** Synthetic | Prompt tested with synthetic data validation: Pattern type, projector-camera geometry |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Pattern type, projector-camera geometry |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Pattern type, projector-camera geometry |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Structured-Light Depth Camera |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Phase unwrapping under defocus, gamma nonlinearity |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Phase unwrapping under defocus, gamma nonlinearity |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Phase unwrapping under defocus, gamma nonlinearity |
| **M3** Real Data | Real experimental data with measured mismatch: Phase unwrapping under defocus, gamma nonlinearity |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Phase unwrapping under defocus, gamma nonlinearity |

### Mismatch Parameters
M→C→D. Gamma [1.5,2.5], extrinsics [0,1] mm/deg, defocus [0,3] px.

### Solvers & Expected Performance
- **Solver**: phase_unwrap

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> C --> D: Estimate gamma curve, projector-camera extrinsics |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate gamma curve, projector-camera extrinsics |
| **M2** Compound | Compound parameter identification (3+ params): Estimate gamma curve, projector-camera extrinsics |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate gamma curve, projector-camera extrinsics |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate gamma curve, projector-camera extrinsics |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct gamma, geometric calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct gamma, geometric calibration |
| **M2** Compound | Compound correction with rho measurement: Correct gamma, geometric calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct gamma, geometric calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct gamma, geometric calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
