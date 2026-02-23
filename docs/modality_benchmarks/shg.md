# Second Harmonic Generation (SHG) Microscopy (`shg`)

**Category**: Microscopy | **Canonical DAG**: M --> R --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: pnp_hqs

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Excitation wavelength, polarization, NA, detection filter |
| **M1** Synthetic | Prompt tested with synthetic data validation: Excitation wavelength, polarization, NA, detection filter |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Excitation wavelength, polarization, NA, detection filter |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Excitation wavelength, polarization, NA, detection filter |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Second Harmonic Generation (SHG) Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: SHG reconstruction under phase-matching error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: SHG reconstruction under phase-matching error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): SHG reconstruction under phase-matching error |
| **M3** Real Data | Real experimental data with measured mismatch: SHG reconstruction under phase-matching error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: SHG reconstruction under phase-matching error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Phase matching error | 0 | [-5%, 5%] | - |
| Excitation power fluctuation | 0 | [0, 10%] | - |
| Collection NA mismatch | 0 | [-0.1, 0.1] | - |

### Solvers & Expected Performance
- **Solver**: pnp_hqs

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate SHG efficiency, polarization orientation |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate SHG efficiency, polarization orientation |
| **M2** Compound | Compound parameter identification (3+ params): Estimate SHG efficiency, polarization orientation |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate SHG efficiency, polarization orientation |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate SHG efficiency, polarization orientation |

### True-Spec Parameters
Phase matching angle, excitation power, SHG efficiency

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct phase-matching, polarization calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct phase-matching, polarization calibration |
| **M2** Compound | Compound correction with rho measurement: Correct phase-matching, polarization calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct phase-matching, polarization calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct phase-matching, polarization calibration |

### Correction Targets
- **Expected rho**: >= 0.65

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
