# Bioluminescence Tomography (BLT) (`bioluminescence_tomo`)

**Category**: Broader Experimental Science | **Canonical DAG**: Src --> R,P --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: diffusion_inversion

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Source depth, wavelength, tissue optics |
| **M1** Synthetic | Prompt tested with synthetic data validation: Source depth, wavelength, tissue optics |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Source depth, wavelength, tissue optics |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Source depth, wavelength, tissue optics |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Bioluminescence Tomography (BLT) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Diffusion model inversion under optical property error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Diffusion model inversion under optical property error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Diffusion model inversion under optical property error |
| **M3** Real Data | Real experimental data with measured mismatch: Diffusion model inversion under optical property error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Diffusion model inversion under optical property error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Optical property error (mu_a, mu_s') | 0 | [0, 20%] | relative |
| Source depth ambiguity | 0 | [0, 5] | mm |
| Autofluorescence background | 0 | [0, 30%] | - |

### Solvers & Expected Performance
- **Solver**: diffusion_inversion

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Src --> R,P --> D: Estimate source location, tissue absorption/scatter |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate source location, tissue absorption/scatter |
| **M2** Compound | Compound parameter identification (3+ params): Estimate source location, tissue absorption/scatter |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate source location, tissue absorption/scatter |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate source location, tissue absorption/scatter |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct optical model, source localization |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct optical model, source localization |
| **M2** Compound | Compound correction with rho measurement: Correct optical model, source localization |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct optical model, source localization |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct optical model, source localization |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
