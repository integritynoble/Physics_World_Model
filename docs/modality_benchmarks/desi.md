# DESI Mass Spectrometry Imaging (`desi`)

**Category**: Spectroscopy & Spectral Imaging | **Canonical DAG**: S --> D | **Carrier**: Ion
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: mass_image_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Spray solvent, voltage, spatial resolution |
| **M1** Synthetic | Prompt tested with synthetic data validation: Spray solvent, voltage, spatial resolution |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Spray solvent, voltage, spatial resolution |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Spray solvent, voltage, spatial resolution |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for DESI Mass Spectrometry Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Mass image under signal variation, matrix effects |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Mass image under signal variation, matrix effects |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Mass image under signal variation, matrix effects |
| **M3** Real Data | Real experimental data with measured mismatch: Mass image under signal variation, matrix effects |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Mass image under signal variation, matrix effects |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Spray angle error | 0 | [-5, 5] | deg |
| Solvent flow variation | 0 | [0, 15%] | - |
| Ion suppression (matrix effect) | 0 | [0, 50%] | - |
| Spatial resolution degradation | 0 | [0, 50%] | - |

### Solvers & Expected Performance
- **Solver**: mass_image_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for S --> D: Estimate ion suppression, spatial offset |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate ion suppression, spatial offset |
| **M2** Compound | Compound parameter identification (3+ params): Estimate ion suppression, spatial offset |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate ion suppression, spatial offset |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate ion suppression, spatial offset |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct spray normalization, co-registration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct spray normalization, co-registration |
| **M2** Compound | Compound correction with rho measurement: Correct spray normalization, co-registration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct spray normalization, co-registration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct spray normalization, co-registration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
