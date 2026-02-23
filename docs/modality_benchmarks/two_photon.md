# Two-Photon / Multiphoton Microscopy (`two_photon`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: richardson_lucy

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Excitation wavelength, NA, scan pattern, depth |
| **M1** Synthetic | Prompt tested with synthetic data validation: Excitation wavelength, NA, scan pattern, depth |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Excitation wavelength, NA, scan pattern, depth |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Excitation wavelength, NA, scan pattern, depth |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Two-Photon / Multiphoton Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Deconvolution under depth-dependent scattering |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Deconvolution under depth-dependent scattering |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Deconvolution under depth-dependent scattering |
| **M3** Real Data | Real experimental data with measured mismatch: Deconvolution under depth-dependent scattering |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Deconvolution under depth-dependent scattering |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Scattering coeff | 10 | [5, 30] | mm^-1 |
| PSF depth scaling | 1.0 | [0.7, 1.5] | - |
| Excitation attenuation | 0.01 | [0.005, 0.02] | per um |
| Motion artifact | 0 | [0, 5] | um |

### Solvers & Expected Performance
- **Solver**: richardson_lucy

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate scattering coefficient, PSF vs depth |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate scattering coefficient, PSF vs depth |
| **M2** Compound | Compound parameter identification (3+ params): Estimate scattering coefficient, PSF vs depth |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate scattering coefficient, PSF vs depth |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate scattering coefficient, PSF vs depth |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct depth-dependent PSF and attenuation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct depth-dependent PSF and attenuation |
| **M2** Compound | Compound correction with rho measurement: Correct depth-dependent PSF and attenuation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct depth-dependent PSF and attenuation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct depth-dependent PSF and attenuation |

### Correction Targets
- **Expected rho**: >= 0.65

### Improvement Roadmap
Add adaptive optics, in-vivo motion model.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
