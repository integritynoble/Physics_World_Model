# Doppler Ultrasound (`doppler_ultrasound`)

**Category**: Medical Imaging | **Canonical DAG**: P --> D | **Carrier**: Acoustic
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: autocorrelation_estimator

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: PRF, wall filter, velocity range, angle of insonation |
| **M1** Synthetic | Prompt tested with synthetic data validation: PRF, wall filter, velocity range, angle of insonation |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for PRF, wall filter, velocity range, angle of insonation |
| **M3** Real Data | Grounded in real experimental/clinical protocols: PRF, wall filter, velocity range, angle of insonation |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Doppler Ultrasound |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Autocorrelation estimator under aliasing, wall filter error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Autocorrelation estimator under aliasing, wall filter error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Autocorrelation estimator under aliasing, wall filter error |
| **M3** Real Data | Real experimental data with measured mismatch: Autocorrelation estimator under aliasing, wall filter error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Autocorrelation estimator under aliasing, wall filter error |

### Mismatch Parameters
P→D, Acoustic. Flow angle [0,90] deg, PRF aliasing +/-20%, wall filter [20,200] Hz.

### Solvers & Expected Performance
- **Solver**: autocorrelation_estimator

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate flow angle, PRF aliasing threshold, clutter |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate flow angle, PRF aliasing threshold, clutter |
| **M2** Compound | Compound parameter identification (3+ params): Estimate flow angle, PRF aliasing threshold, clutter |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate flow angle, PRF aliasing threshold, clutter |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate flow angle, PRF aliasing threshold, clutter |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct angle, anti-aliasing, clutter filter |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct angle, anti-aliasing, clutter filter |
| **M2** Compound | Compound correction with rho measurement: Correct angle, anti-aliasing, clutter filter |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct angle, anti-aliasing, clutter filter |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct angle, anti-aliasing, clutter filter |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
