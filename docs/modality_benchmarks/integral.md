# Integral Photography (`integral`)

**Category**: Computational Optics | **Canonical DAG**: C --> S --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: depth_estimation

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Lens array geometry, baseline, depth range |
| **M1** Synthetic | Prompt tested with synthetic data validation: Lens array geometry, baseline, depth range |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Lens array geometry, baseline, depth range |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Lens array geometry, baseline, depth range |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Integral Photography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Depth estimation under lens distortion |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Depth estimation under lens distortion |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Depth estimation under lens distortion |
| **M3** Real Data | Real experimental data with measured mismatch: Depth estimation under lens distortion |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Depth estimation under lens distortion |

### Mismatch Parameters
C→S→D. Lens position [0,0.5] mm, distortion [0,3%].

### Solvers & Expected Performance
- **Solver**: depth_estimation

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> S --> D: Estimate lens positions, distortion coefficients |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate lens positions, distortion coefficients |
| **M2** Compound | Compound parameter identification (3+ params): Estimate lens positions, distortion coefficients |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate lens positions, distortion coefficients |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate lens positions, distortion coefficients |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct geometric calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct geometric calibration |
| **M2** Compound | Compound correction with rho measurement: Correct geometric calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct geometric calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct geometric calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
