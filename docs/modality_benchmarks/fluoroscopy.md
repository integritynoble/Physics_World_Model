# Fluoroscopy (`fluoroscopy`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: tv_fista

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Frame rate, dose per frame, detector type |
| **M1** Synthetic | Prompt tested with synthetic data validation: Frame rate, dose per frame, detector type |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Frame rate, dose per frame, detector type |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Frame rate, dose per frame, detector type |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Fluoroscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: TV-FISTA under temporal lag, scatter, geometric distortion |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: TV-FISTA under temporal lag, scatter, geometric distortion |
| **M2** Compound | Compound mismatch (3+ params simultaneously): TV-FISTA under temporal lag, scatter, geometric distortion |
| **M3** Real Data | Real experimental data with measured mismatch: TV-FISTA under temporal lag, scatter, geometric distortion |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: TV-FISTA under temporal lag, scatter, geometric distortion |

### Mismatch Parameters
Pi → D, X-ray. Lag [0,0.15], pincushion [0,3%], veiling glare [0,10%].

### Solvers & Expected Performance
- **Solver**: tv_fista

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate lag coefficient, scatter model, pincushion distortion |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate lag coefficient, scatter model, pincushion distortion |
| **M2** Compound | Compound parameter identification (3+ params): Estimate lag coefficient, scatter model, pincushion distortion |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate lag coefficient, scatter model, pincushion distortion |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate lag coefficient, scatter model, pincushion distortion |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct lag, flat-field, geometric distortion |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct lag, flat-field, geometric distortion |
| **M2** Compound | Compound correction with rho measurement: Correct lag, flat-field, geometric distortion |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct lag, flat-field, geometric distortion |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct lag, flat-field, geometric distortion |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
