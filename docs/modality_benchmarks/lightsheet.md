# Light-Sheet Fluorescence Microscopy (LSFM) (`lightsheet`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M1 | **Forward Model**: linear_operator | **Default Solver**: fourier_notch_destripe

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Sheet thickness, detection NA, multi-view config |
| **M1** Synthetic | Prompt tested with synthetic data validation: Sheet thickness, detection NA, multi-view config |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Sheet thickness, detection NA, multi-view config |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Sheet thickness, detection NA, multi-view config |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Light-Sheet Fluorescence Microscopy (LSFM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Destripe + deconvolution + multi-view fusion |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Destripe + deconvolution + multi-view fusion |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Destripe + deconvolution + multi-view fusion |
| **M3** Real Data | Real experimental data with measured mismatch: Destripe + deconvolution + multi-view fusion |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Destripe + deconvolution + multi-view fusion |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Sheet thickness | 5.0 | [2.0, 15.0] | um |
| Sheet tilt | 0 | [-3, 3] | deg |
| Stripe strength | 0.2 | [0, 0.8] | relative |
| Attenuation coeff | 0.02 | [0.005, 0.08] | per slice |

### Solvers & Expected Performance
- **Solver**: fourier_notch_destripe

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate stripe strength, sheet profile, tilt |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate stripe strength, sheet profile, tilt |
| **M2** Compound | Compound parameter identification (3+ params): Estimate stripe strength, sheet profile, tilt |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate stripe strength, sheet profile, tilt |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate stripe strength, sheet profile, tilt |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct sheet alignment, remove stripe artifacts |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct sheet alignment, remove stripe artifacts |
| **M2** Compound | Compound correction with rho measurement: Correct sheet alignment, remove stripe artifacts |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct sheet alignment, remove stripe artifacts |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct sheet alignment, remove stripe artifacts |

### Correction Targets
- **Expected rho**: >= 0.75

### Improvement Roadmap
Add scattering-induced stripe, tile stitching error.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
