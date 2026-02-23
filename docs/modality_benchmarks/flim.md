# Fluorescence Lifetime Imaging (FLIM) (`flim`)

**Category**: Microscopy | **Canonical DAG**: M --> R --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: phasor

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Excitation pulse, time gates, lifetime range |
| **M1** Synthetic | Prompt tested with synthetic data validation: Excitation pulse, time gates, lifetime range |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Excitation pulse, time gates, lifetime range |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Excitation pulse, time gates, lifetime range |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Fluorescence Lifetime Imaging (FLIM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Phasor analysis under IRF mismatch |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Phasor analysis under IRF mismatch |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Phasor analysis under IRF mismatch |
| **M3** Real Data | Real experimental data with measured mismatch: Phasor analysis under IRF mismatch |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Phasor analysis under IRF mismatch |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| IRF width | 80 | [40, 200] | ps |
| IRF shift | 0 | [-50, 50] | ps |
| Afterpulsing | 0.01 | [0, 0.1] | relative |
| Pile-up fraction | 0 | [0, 0.05] | - |

### Solvers & Expected Performance
- **Solver**: phasor

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate IRF width, background, afterpulsing |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate IRF width, background, afterpulsing |
| **M2** Compound | Compound parameter identification (3+ params): Estimate IRF width, background, afterpulsing |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate IRF width, background, afterpulsing |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate IRF width, background, afterpulsing |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct IRF, calibrate lifetime axis |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct IRF, calibrate lifetime axis |
| **M2** Compound | Compound correction with rho measurement: Correct IRF, calibrate lifetime axis |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct IRF, calibrate lifetime axis |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct IRF, calibrate lifetime axis |

### Correction Targets
- **Expected rho**: >= 0.70

### Improvement Roadmap
Add multi-exponential, FRET efficiency benchmark.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
