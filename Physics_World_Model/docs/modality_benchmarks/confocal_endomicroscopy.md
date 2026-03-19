# Confocal Laser Endomicroscopy (CLE) (`confocal_endomicroscopy`)

**Category**: Medical Imaging | **Canonical DAG**: M --> C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: fiber_deconvolution

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Fiber bundle, laser wavelength, frame rate |
| **M1** Synthetic | Prompt tested with synthetic data validation: Fiber bundle, laser wavelength, frame rate |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Fiber bundle, laser wavelength, frame rate |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Fiber bundle, laser wavelength, frame rate |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Confocal Laser Endomicroscopy (CLE) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Deconvolution under fiber honeycomb pattern |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Deconvolution under fiber honeycomb pattern |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Deconvolution under fiber honeycomb pattern |
| **M3** Real Data | Real experimental data with measured mismatch: Deconvolution under fiber honeycomb pattern |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Deconvolution under fiber honeycomb pattern |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Fiber bundle honeycomb pattern | measured | +/- 5% pitch | - |
| Motion artifact | 0 | [0, 10] | px/frame |
| Fluorescein concentration variation | 1.0 | [0.3, 3.0] | relative |

### Solvers & Expected Performance
- **Solver**: fiber_deconvolution

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> C --> D: Estimate fiber core positions, coupling efficiency |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate fiber core positions, coupling efficiency |
| **M2** Compound | Compound parameter identification (3+ params): Estimate fiber core positions, coupling efficiency |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate fiber core positions, coupling efficiency |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate fiber core positions, coupling efficiency |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct fiber pattern, interpolation artifacts |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct fiber pattern, interpolation artifacts |
| **M2** Compound | Compound correction with rho measurement: Correct fiber pattern, interpolation artifacts |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct fiber pattern, interpolation artifacts |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct fiber pattern, interpolation artifacts |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
