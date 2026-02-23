# Neutron Diffraction (`neutron_diffraction`)

**Category**: Scientific Instrumentation | **Canonical DAG**: R --> S --> D | **Carrier**: Neutron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: rietveld_refinement

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Wavelength, detector coverage, sample environment |
| **M1** Synthetic | Prompt tested with synthetic data validation: Wavelength, detector coverage, sample environment |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Wavelength, detector coverage, sample environment |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Wavelength, detector coverage, sample environment |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Neutron Diffraction |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Rietveld/Pawley under background, absorption, extinction |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Rietveld/Pawley under background, absorption, extinction |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Rietveld/Pawley under background, absorption, extinction |
| **M3** Real Data | Real experimental data with measured mismatch: Rietveld/Pawley under background, absorption, extinction |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Rietveld/Pawley under background, absorption, extinction |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Wavelength calibration | 0 | [-0.1%, 0.1%] | - |
| Absorption correction | 0 | [0, 10%] | - |
| Texture/preferred orientation | none | [0, 20%] | - |
| TOF frame overlap | 0 | [0, 5%] of peaks | - |

### Solvers & Expected Performance
- **Solver**: rietveld_refinement

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for R --> S --> D: Estimate structure parameters, absorption correction |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate structure parameters, absorption correction |
| **M2** Compound | Compound parameter identification (3+ params): Estimate structure parameters, absorption correction |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate structure parameters, absorption correction |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate structure parameters, absorption correction |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct extinction, absorption, background |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct extinction, absorption, background |
| **M2** Compound | Compound correction with rho measurement: Correct extinction, absorption, background |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct extinction, absorption, background |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct extinction, absorption, background |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
