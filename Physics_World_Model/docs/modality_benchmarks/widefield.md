# Widefield Fluorescence Microscopy (`widefield`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M3 | **Forward Model**: linear_operator | **Default Solver**: richardson_lucy

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | "Design widefield fluorescence for GFP-labeled fixed cells: 60x oil, NA 1.4, emission 500-550 nm, pixel 100 nm, FOV 80 um, sCMOS." |
| **M1** Synthetic | Prompt tested with synthetic data validation: PSF design, NA selection, illumination uniformity |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for PSF design, NA selection, illumination uniformity |
| **M3** Real Data | Grounded in real experimental/clinical protocols: PSF design, NA selection, illumination uniformity |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Widefield Fluorescence Microscopy |

### Design Parameters
| Parameter | Range | Unit |
|-----------|-------|------|
| Objective NA | 0.4 - 1.49 | - |
| PSF sigma (lateral) | 0.8 - 4.0 | px |
| Emission wavelength | 400 - 800 | nm |
| Read noise | 1.0 - 10.0 | e- |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Deconvolution under PSF mismatch and Poisson noise |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Deconvolution under PSF mismatch and Poisson noise |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Deconvolution under PSF mismatch and Poisson noise |
| **M3** Real Data | Real experimental data with measured mismatch: Deconvolution under PSF mismatch and Poisson noise |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Deconvolution under PSF mismatch and Poisson noise |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| PSF sigma | 2.0 | [1.2, 3.5] | px |
| Background level | 50 | [0, 200] | counts |
| Gain | 1.0 | [0.85, 1.15] | - |
| Flatfield non-uniformity | 0% | [0%, 15%] | peak-to-peak |
| Photobleaching rate | 0 | [0, 0.05] | per frame |

### Solvers & Expected Performance
- **Solver(s)**: Richardson-Lucy, PnP-HQS, Wiener
- **Scenario I PSNR**: 30-38 dB
- **Scenario II drop**: 1-5 dB

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate PSF sigma, background level, gain |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate PSF sigma, background level, gain |
| **M2** Compound | Compound parameter identification (3+ params): Estimate PSF sigma, background level, gain |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate PSF sigma, background level, gain |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate PSF sigma, background level, gain |

### True-Spec Parameters
PSF sigma_x (2.13), sigma_y (2.07), background (47.3), read noise (5.8), gain (1.03).

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct PSF model, suppress out-of-focus blur |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct PSF model, suppress out-of-focus blur |
| **M2** Compound | Compound correction with rho measurement: Correct PSF model, suppress out-of-focus blur |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct PSF model, suppress out-of-focus blur |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct PSF model, suppress out-of-focus blur |

### Correction Targets
- **Expected rho**: >= 0.85
- **PSNR gain**: +1 to +5 dB

### Improvement Roadmap
Add compound mismatch (PSF + background + flatfield), depth-dependent PSF.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
