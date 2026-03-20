# Terahertz Imaging (THz) (`terahertz`)

**Category**: Industrial Inspection | **Canonical DAG**: P --> D | **Carrier**: THz photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: deconvolution

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Frequency range, imaging mode, spatial resolution |
| **M1** Synthetic | Prompt tested with synthetic data validation: Frequency range, imaging mode, spatial resolution |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Frequency range, imaging mode, spatial resolution |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Frequency range, imaging mode, spatial resolution |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Terahertz Imaging (THz) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Deconvolution under water vapor absorption, etalon |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Deconvolution under water vapor absorption, etalon |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Deconvolution under water vapor absorption, etalon |
| **M3** Real Data | Real experimental data with measured mismatch: Deconvolution under water vapor absorption, etalon |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Deconvolution under water vapor absorption, etalon |

### Mismatch Parameters
P→D. Water vapor [0,5] dB, etalon [0,100] GHz, RI +/-5%.

### Solvers & Expected Performance
- **Solver**: deconvolution

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate thickness, refractive index, absorption |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate thickness, refractive index, absorption |
| **M2** Compound | Compound parameter identification (3+ params): Estimate thickness, refractive index, absorption |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate thickness, refractive index, absorption |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate thickness, refractive index, absorption |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct etalon artifacts, vapor lines |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct etalon artifacts, vapor lines |
| **M2** Compound | Compound correction with rho measurement: Correct etalon artifacts, vapor lines |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct etalon artifacts, vapor lines |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct etalon artifacts, vapor lines |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
