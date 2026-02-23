# FTIR Spectroscopic Imaging (`ftir_imaging`)

**Category**: Spectroscopy & Spectral Imaging | **Canonical DAG**: M --> Sigma --> D | **Carrier**: IR photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: interferogram_fft

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Spectral range, resolution, interferometer type |
| **M1** Synthetic | Prompt tested with synthetic data validation: Spectral range, resolution, interferometer type |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Spectral range, resolution, interferometer type |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Spectral range, resolution, interferometer type |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for FTIR Spectroscopic Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Interferogram processing under apodization, phase error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Interferogram processing under apodization, phase error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Interferogram processing under apodization, phase error |
| **M3** Real Data | Real experimental data with measured mismatch: Interferogram processing under apodization, phase error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Interferogram processing under apodization, phase error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Wavenumber calibration | 0 | [-2, 2] | cm^-1 |
| Water vapor absorption | 0 | [0, variable] | - |
| Detector nonlinearity | 0 | [0, 5%] | - |
| ATR crystal RI error | 0 | [-1%, 1%] | - |

### Solvers & Expected Performance
- **Solver**: interferogram_fft

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> Sigma --> D: Estimate phase error, wavenumber calibration |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate phase error, wavenumber calibration |
| **M2** Compound | Compound parameter identification (3+ params): Estimate phase error, wavenumber calibration |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate phase error, wavenumber calibration |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate phase error, wavenumber calibration |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct phase, apodization, atmospheric absorption |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct phase, apodization, atmospheric absorption |
| **M2** Compound | Compound correction with rho measurement: Correct phase, apodization, atmospheric absorption |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct phase, apodization, atmospheric absorption |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct phase, apodization, atmospheric absorption |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
