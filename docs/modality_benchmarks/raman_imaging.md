# Raman Imaging / Microscopy (`raman_imaging`)

**Category**: Spectroscopy & Spectral Imaging | **Canonical DAG**: M --> R --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: spectral_unmixing

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | "Design confocal Raman for pharmaceutical tablet: 785 nm, 10 um resolution, 100-3200 cm^-1." |
| **M1** Synthetic | Prompt tested with synthetic data validation: Excitation wavelength, grating, integration time |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Excitation wavelength, grating, integration time |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Excitation wavelength, grating, integration time |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Raman Imaging / Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Spectral unmixing under fluorescence background, cosmic rays |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Spectral unmixing under fluorescence background, cosmic rays |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Spectral unmixing under fluorescence background, cosmic rays |
| **M3** Real Data | Real experimental data with measured mismatch: Spectral unmixing under fluorescence background, cosmic rays |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Spectral unmixing under fluorescence background, cosmic rays |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Spectral calibration shift | 0 | [-2, 2] | cm^-1 |
| Fluorescence background | 0 | [0, 10x Raman signal] | relative |
| Laser power fluctuation | 0 | [0, 5%] | - |
| Cosmic ray artifact | 0 | [0, 1%] of spectra | - |

### Solvers & Expected Performance
- **Solver**: spectral_unmixing

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate background, peak positions, linewidths |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate background, peak positions, linewidths |
| **M2** Compound | Compound parameter identification (3+ params): Estimate background, peak positions, linewidths |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate background, peak positions, linewidths |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate background, peak positions, linewidths |

### True-Spec Parameters
Spectral calibration, laser power, fluorescence model, cosmic ray locations

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct fluorescence baseline, cosmic ray removal |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct fluorescence baseline, cosmic ray removal |
| **M2** Compound | Compound correction with rho measurement: Correct fluorescence baseline, cosmic ray removal |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct fluorescence baseline, cosmic ray removal |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct fluorescence baseline, cosmic ray removal |

### Correction Targets
- **Expected rho**: >= 0.75

### Improvement Roadmap
Add SERS benchmark, baseline subtraction comparison.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
