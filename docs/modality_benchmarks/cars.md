# Coherent Anti-Stokes Raman (CARS) Microscopy (`cars`)

**Category**: Spectroscopy & Spectral Imaging | **Canonical DAG**: M --> R --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: nrb_removal

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Pump/Stokes wavelengths, bandwidth, delay |
| **M1** Synthetic | Prompt tested with synthetic data validation: Pump/Stokes wavelengths, bandwidth, delay |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Pump/Stokes wavelengths, bandwidth, delay |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Pump/Stokes wavelengths, bandwidth, delay |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Coherent Anti-Stokes Raman (CARS) Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: CARS recon under non-resonant background |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: CARS recon under non-resonant background |
| **M2** Compound | Compound mismatch (3+ params simultaneously): CARS recon under non-resonant background |
| **M3** Real Data | Real experimental data with measured mismatch: CARS recon under non-resonant background |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: CARS recon under non-resonant background |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Pump-Stokes frequency offset | 0 | [-5, 5] | cm^-1 |
| Non-resonant background | 0 | [0, 50%] of signal | - |
| Chirp mismatch | 0 | [0, 500] | fs^2 |

### Solvers & Expected Performance
- **Solver**: nrb_removal

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate non-resonant background, spectral phase |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate non-resonant background, spectral phase |
| **M2** Compound | Compound parameter identification (3+ params): Estimate non-resonant background, spectral phase |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate non-resonant background, spectral phase |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate non-resonant background, spectral phase |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct non-resonant background, phase retrieval |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct non-resonant background, phase retrieval |
| **M2** Compound | Compound correction with rho measurement: Correct non-resonant background, phase retrieval |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct non-resonant background, phase retrieval |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct non-resonant background, phase retrieval |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
