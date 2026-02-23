# Stellar Coronagraphy (`coronagraphy`)

**Category**: Astronomy & Space Imaging | **Canonical DAG**: M --> P --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: adi_speckle_subtraction

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | "Design stellar coronagraph for exoplanet imaging: Lyot stop, 1e-8 contrast, 3 lambda/D IWA." |
| **M1** Synthetic | Prompt tested with synthetic data validation: Coronagraph type (Lyot/vortex), IWA, contrast ratio |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Coronagraph type (Lyot/vortex), IWA, contrast ratio |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Coronagraph type (Lyot/vortex), IWA, contrast ratio |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Stellar Coronagraphy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Post-processing under speckle residuals, wind-driven halo |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Post-processing under speckle residuals, wind-driven halo |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Post-processing under speckle residuals, wind-driven halo |
| **M3** Real Data | Real experimental data with measured mismatch: Post-processing under speckle residuals, wind-driven halo |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Post-processing under speckle residuals, wind-driven halo |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Coronagraph mask centering | 0 | [0, 0.1] | lambda/D |
| Wavefront error (WFE) | 0 | [0, lambda/100] rms | - |
| Stellar leakage | 1e-6 | [1e-7, 1e-4] | contrast |
| Speckle lifetime | static | [0.1, 100] | s |

### Solvers & Expected Performance
- **Solver**: adi_speckle_subtraction

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> P --> D: Estimate speckle field, quasi-static aberration |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate speckle field, quasi-static aberration |
| **M2** Compound | Compound parameter identification (3+ params): Estimate speckle field, quasi-static aberration |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate speckle field, quasi-static aberration |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate speckle field, quasi-static aberration |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct speckle subtraction (ADI/SDI), wavefront |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct speckle subtraction (ADI/SDI), wavefront |
| **M2** Compound | Compound correction with rho measurement: Correct speckle subtraction (ADI/SDI), wavefront |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct speckle subtraction (ADI/SDI), wavefront |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct speckle subtraction (ADI/SDI), wavefront |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Post-processing comparison (ADI, SDI, RDI); wavefront sensing & control loop.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
