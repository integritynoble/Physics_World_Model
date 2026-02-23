# Diffuse Optical Tomography (DOT) (`dot`)

**Category**: Medical Imaging | **Canonical DAG**: M --> R,P,R --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: born_approx

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Source-detector layout, wavelength selection, time/frequency domain |
| **M1** Synthetic | Prompt tested with synthetic data validation: Source-detector layout, wavelength selection, time/frequency domain |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Source-detector layout, wavelength selection, time/frequency domain |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Source-detector layout, wavelength selection, time/frequency domain |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Diffuse Optical Tomography (DOT) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Born approximation inversion under scattering coefficient error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Born approximation inversion under scattering coefficient error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Born approximation inversion under scattering coefficient error |
| **M3** Real Data | Real experimental data with measured mismatch: Born approximation inversion under scattering coefficient error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Born approximation inversion under scattering coefficient error |

### Mismatch Parameters
M → R,P,R → D, Photon. mu_a [0.005,0.05], mu_s' [0.5,2.0], coupling [0.5,1.5].

### Solvers & Expected Performance
- **Solver**: born_approx

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R,P,R --> D: Estimate absorption/scattering coefficients, boundary conditions |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate absorption/scattering coefficients, boundary conditions |
| **M2** Compound | Compound parameter identification (3+ params): Estimate absorption/scattering coefficients, boundary conditions |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate absorption/scattering coefficients, boundary conditions |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate absorption/scattering coefficients, boundary conditions |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct optical properties, boundary model |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct optical properties, boundary model |
| **M2** Compound | Compound correction with rho measurement: Correct optical properties, boundary model |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct optical properties, boundary model |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct optical properties, boundary model |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
