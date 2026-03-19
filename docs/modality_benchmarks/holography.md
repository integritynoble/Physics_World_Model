# Digital Holographic Microscopy (`holography`)

**Category**: Coherent Imaging | **Canonical DAG**: P --> D | **Carrier**: Photon
**Current Maturity**: M1 | **Forward Model**: nonlinear_operator | **Default Solver**: angular_spectrum

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Reference beam angle, wavelength, off-axis vs inline |
| **M1** Synthetic | Prompt tested with synthetic data validation: Reference beam angle, wavelength, off-axis vs inline |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Reference beam angle, wavelength, off-axis vs inline |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Reference beam angle, wavelength, off-axis vs inline |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Digital Holographic Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Angular spectrum under reference beam angle error, vibration |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Angular spectrum under reference beam angle error, vibration |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Angular spectrum under reference beam angle error, vibration |
| **M3** Real Data | Real experimental data with measured mismatch: Angular spectrum under reference beam angle error, vibration |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Angular spectrum under reference beam angle error, vibration |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Reference angle error | 0 | [-1, 1] | deg |
| Carrier frequency error | 0 | [-5%, 5%] | - |
| Vibration | 0 | [0, lambda/10] | - |

### Solvers & Expected Performance
- **Solver**: angular_spectrum

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate carrier frequency, reference angle, phase offset |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate carrier frequency, reference angle, phase offset |
| **M2** Compound | Compound parameter identification (3+ params): Estimate carrier frequency, reference angle, phase offset |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate carrier frequency, reference angle, phase offset |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate carrier frequency, reference angle, phase offset |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct reference beam model, vibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct reference beam model, vibration |
| **M2** Compound | Compound correction with rho measurement: Correct reference beam model, vibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct reference beam model, vibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct reference beam model, vibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
