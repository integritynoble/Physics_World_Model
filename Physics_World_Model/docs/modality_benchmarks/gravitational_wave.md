# Gravitational Wave Detection (`gravitational_wave`)

**Category**: Broader Experimental Science | **Canonical DAG**: P --> Sigma --> D | **Carrier**: Gravitational
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: matched_filter

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Arm length, laser power, mirror quality |
| **M1** Synthetic | Prompt tested with synthetic data validation: Arm length, laser power, mirror quality |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Arm length, laser power, mirror quality |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Arm length, laser power, mirror quality |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Gravitational Wave Detection |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Matched filter under noise non-stationarity, glitches |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Matched filter under noise non-stationarity, glitches |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Matched filter under noise non-stationarity, glitches |
| **M3** Real Data | Real experimental data with measured mismatch: Matched filter under noise non-stationarity, glitches |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Matched filter under noise non-stationarity, glitches |

### Mismatch Parameters
P→Sigma→D. Calibration +/-5%, PSD +/-10%, glitch [0,1]/100s.

### Solvers & Expected Performance
- **Solver**: matched_filter

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> Sigma --> D: Estimate noise PSD, glitch morphology |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate noise PSD, glitch morphology |
| **M2** Compound | Compound parameter identification (3+ params): Estimate noise PSD, glitch morphology |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate noise PSD, glitch morphology |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate noise PSD, glitch morphology |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct calibration, glitch subtraction |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct calibration, glitch subtraction |
| **M2** Compound | Compound correction with rho measurement: Correct calibration, glitch subtraction |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct calibration, glitch subtraction |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct calibration, glitch subtraction |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
