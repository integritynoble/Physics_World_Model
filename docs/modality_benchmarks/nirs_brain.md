# Functional Near-Infrared Spectroscopy (fNIRS) (`nirs_brain`)

**Category**: Medical Imaging | **Canonical DAG**: M --> R,P --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: modified_beer_lambert

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Source-detector distance, wavelengths, sampling rate |
| **M1** Synthetic | Prompt tested with synthetic data validation: Source-detector distance, wavelengths, sampling rate |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Source-detector distance, wavelengths, sampling rate |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Source-detector distance, wavelengths, sampling rate |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Functional Near-Infrared Spectroscopy (fNIRS) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Modified Beer-Lambert under scalp coupling variation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Modified Beer-Lambert under scalp coupling variation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Modified Beer-Lambert under scalp coupling variation |
| **M3** Real Data | Real experimental data with measured mismatch: Modified Beer-Lambert under scalp coupling variation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Modified Beer-Lambert under scalp coupling variation |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Source-detector coupling | 1.0 | [0.5, 1.5] per optode | - |
| Scalp-brain distance variation | 0 | [0, 5] | mm |
| Motion artifact (head) | 0 | [0, 10%] signal | - |
| Systemic physiology contamination | 0 | [0, 30%] | - |

### Solvers & Expected Performance
- **Solver**: modified_beer_lambert

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R,P --> D: Estimate DPF, coupling coefficients, motion artifact |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate DPF, coupling coefficients, motion artifact |
| **M2** Compound | Compound parameter identification (3+ params): Estimate DPF, coupling coefficients, motion artifact |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate DPF, coupling coefficients, motion artifact |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate DPF, coupling coefficients, motion artifact |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct motion, coupling, superficial signal |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct motion, coupling, superficial signal |
| **M2** Compound | Compound correction with rho measurement: Correct motion, coupling, superficial signal |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct motion, coupling, superficial signal |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct motion, coupling, superficial signal |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
