# Stimulated Raman Scattering (SRS) Microscopy (`srs`)

**Category**: Spectroscopy & Spectral Imaging | **Canonical DAG**: M --> R --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: lock_in_demod

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Modulation frequency, pump/Stokes power, lock-in |
| **M1** Synthetic | Prompt tested with synthetic data validation: Modulation frequency, pump/Stokes power, lock-in |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Modulation frequency, pump/Stokes power, lock-in |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Modulation frequency, pump/Stokes power, lock-in |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Stimulated Raman Scattering (SRS) Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: SRS imaging under photothermal artifact, cross-phase mod |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: SRS imaging under photothermal artifact, cross-phase mod |
| **M2** Compound | Compound mismatch (3+ params simultaneously): SRS imaging under photothermal artifact, cross-phase mod |
| **M3** Real Data | Real experimental data with measured mismatch: SRS imaging under photothermal artifact, cross-phase mod |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: SRS imaging under photothermal artifact, cross-phase mod |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Lock-in phase error | 0 | [-10, 10] | deg |
| Cross-phase modulation | 0 | [0, 5%] | - |
| Laser intensity noise (RIN) | -150 | [-140, -160] | dBc/Hz |

### Solvers & Expected Performance
- **Solver**: lock_in_demod

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate photothermal contribution, XPM artifacts |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate photothermal contribution, XPM artifacts |
| **M2** Compound | Compound parameter identification (3+ params): Estimate photothermal contribution, XPM artifacts |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate photothermal contribution, XPM artifacts |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate photothermal contribution, XPM artifacts |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct photothermal, XPM background |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct photothermal, XPM background |
| **M2** Compound | Compound correction with rho measurement: Correct photothermal, XPM background |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct photothermal, XPM background |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct photothermal, XPM background |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
