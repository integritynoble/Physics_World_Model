# Secondary Ion Mass Spectrometry (SIMS) Imaging (`sims`)

**Category**: Spectroscopy & Spectral Imaging | **Canonical DAG**: S --> D | **Carrier**: Ion
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: mass_image_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Primary ion species, energy, mass analyzer |
| **M1** Synthetic | Prompt tested with synthetic data validation: Primary ion species, energy, mass analyzer |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Primary ion species, energy, mass analyzer |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Primary ion species, energy, mass analyzer |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Secondary Ion Mass Spectrometry (SIMS) Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Mass image under dead time, mass interference |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Mass image under dead time, mass interference |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Mass image under dead time, mass interference |
| **M3** Real Data | Real experimental data with measured mismatch: Mass image under dead time, mass interference |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Mass image under dead time, mass interference |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Mass calibration drift | 0 | [-5, 5] | ppm |
| Matrix effect (sputter yield) | 0 | [0, 50%] | - |
| Crater edge effect | 0 | [0, 10%] of area | - |
| Charging (insulating samples) | 0 | [0, 200] | V |

### Solvers & Expected Performance
- **Solver**: mass_image_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for S --> D: Estimate dead time, isotope ratios, matrix effect |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate dead time, isotope ratios, matrix effect |
| **M2** Compound | Compound parameter identification (3+ params): Estimate dead time, isotope ratios, matrix effect |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate dead time, isotope ratios, matrix effect |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate dead time, isotope ratios, matrix effect |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct dead time, mass calibration, matrix |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct dead time, mass calibration, matrix |
| **M2** Compound | Compound correction with rho measurement: Correct dead time, mass calibration, matrix |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct dead time, mass calibration, matrix |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct dead time, mass calibration, matrix |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
