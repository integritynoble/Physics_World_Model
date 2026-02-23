# Passive Microwave Radiometry (`passive_microwave`)

**Category**: Remote Sensing | **Canonical DAG**: Sigma --> D | **Carrier**: RF
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: deconvolution

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Frequency, spatial resolution, integration time |
| **M1** Synthetic | Prompt tested with synthetic data validation: Frequency, spatial resolution, integration time |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Frequency, spatial resolution, integration time |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Frequency, spatial resolution, integration time |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Passive Microwave Radiometry |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Deconvolution under antenna pattern error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Deconvolution under antenna pattern error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Deconvolution under antenna pattern error |
| **M3** Real Data | Real experimental data with measured mismatch: Deconvolution under antenna pattern error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Deconvolution under antenna pattern error |

### Mismatch Parameters
Sigma→D, RF. Antenna pattern +/-5%, gain drift +/-1%.

### Solvers & Expected Performance
- **Solver**: deconvolution

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Sigma --> D: Estimate antenna pattern, gain calibration |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate antenna pattern, gain calibration |
| **M2** Compound | Compound parameter identification (3+ params): Estimate antenna pattern, gain calibration |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate antenna pattern, gain calibration |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate antenna pattern, gain calibration |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct antenna pattern, radiometric cal |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct antenna pattern, radiometric cal |
| **M2** Compound | Compound correction with rho measurement: Correct antenna pattern, radiometric cal |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct antenna pattern, radiometric cal |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct antenna pattern, radiometric cal |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
