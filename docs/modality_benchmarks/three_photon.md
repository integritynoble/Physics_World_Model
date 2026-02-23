# Three-Photon Microscopy (`three_photon`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: richardson_lucy

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Excitation 1300/1700 nm, repetition rate, depth |
| **M1** Synthetic | Prompt tested with synthetic data validation: Excitation 1300/1700 nm, repetition rate, depth |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Excitation 1300/1700 nm, repetition rate, depth |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Excitation 1300/1700 nm, repetition rate, depth |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Three-Photon Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Deconvolution under deep-tissue scattering |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Deconvolution under deep-tissue scattering |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Deconvolution under deep-tissue scattering |
| **M3** Real Data | Real experimental data with measured mismatch: Deconvolution under deep-tissue scattering |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Deconvolution under deep-tissue scattering |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Scattering coeff | 15 | [8, 40] | mm^-1 |
| Excitation wavelength shift | 0 | [-10, 10] | nm |
| Depth-dependent PSF | varies | scale [0.5, 2.0] | - |

### Solvers & Expected Performance
- **Solver**: richardson_lucy

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate scattering length, pulse broadening |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate scattering length, pulse broadening |
| **M2** Compound | Compound parameter identification (3+ params): Estimate scattering length, pulse broadening |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate scattering length, pulse broadening |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate scattering length, pulse broadening |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct depth-dependent attenuation and PSF |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct depth-dependent attenuation and PSF |
| **M2** Compound | Compound correction with rho measurement: Correct depth-dependent attenuation and PSF |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct depth-dependent attenuation and PSF |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct depth-dependent attenuation and PSF |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Test imaging depth 1-2 mm in brain tissue; compare vs two-photon.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
