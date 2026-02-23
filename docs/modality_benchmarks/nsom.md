# Near-field Scanning Optical Microscopy (NSOM) (`nsom`)

**Category**: Scanning Probe Microscopy | **Canonical DAG**: M --> C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: near_field_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Aperture size, feedback, wavelength, illumination mode |
| **M1** Synthetic | Prompt tested with synthetic data validation: Aperture size, feedback, wavelength, illumination mode |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Aperture size, feedback, wavelength, illumination mode |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Aperture size, feedback, wavelength, illumination mode |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Near-field Scanning Optical Microscopy (NSOM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Near-field recon under topographic artifact, coupling |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Near-field recon under topographic artifact, coupling |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Near-field recon under topographic artifact, coupling |
| **M3** Real Data | Real experimental data with measured mismatch: Near-field recon under topographic artifact, coupling |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Near-field recon under topographic artifact, coupling |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Tip-sample distance | 10 | [5, 50] | nm |
| Aperture size error | 0 | [-20%, 20%] | - |
| Topographic coupling | 0 | [0, 30%] | - |
| Far-field background | 0 | [0, 20%] | - |

### Solvers & Expected Performance
- **Solver**: near_field_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> C --> D: Estimate topographic crosstalk, far-field background |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate topographic crosstalk, far-field background |
| **M2** Compound | Compound parameter identification (3+ params): Estimate topographic crosstalk, far-field background |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate topographic crosstalk, far-field background |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate topographic crosstalk, far-field background |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct topographic artifact, background subtraction |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct topographic artifact, background subtraction |
| **M2** Compound | Compound correction with rho measurement: Correct topographic artifact, background subtraction |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct topographic artifact, background subtraction |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct topographic artifact, background subtraction |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
