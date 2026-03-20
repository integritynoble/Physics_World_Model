# Fundus Camera (`fundus`)

**Category**: Medical Imaging | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: richardson_lucy

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: FOV, illumination wavelength, mydriasis |
| **M1** Synthetic | Prompt tested with synthetic data validation: FOV, illumination wavelength, mydriasis |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for FOV, illumination wavelength, mydriasis |
| **M3** Real Data | Grounded in real experimental/clinical protocols: FOV, illumination wavelength, mydriasis |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Fundus Camera |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Richardson-Lucy under aberration, non-uniform illumination |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Richardson-Lucy under aberration, non-uniform illumination |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Richardson-Lucy under aberration, non-uniform illumination |
| **M3** Real Data | Real experimental data with measured mismatch: Richardson-Lucy under aberration, non-uniform illumination |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Richardson-Lucy under aberration, non-uniform illumination |

### Mismatch Parameters
C→D, Photon. Aberration [0,0.5] waves, illumination [0,30%].

### Solvers & Expected Performance
- **Solver**: richardson_lucy

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate aberration coefficients, illumination profile |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate aberration coefficients, illumination profile |
| **M2** Compound | Compound parameter identification (3+ params): Estimate aberration coefficients, illumination profile |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate aberration coefficients, illumination profile |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate aberration coefficients, illumination profile |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct aberrations, flat-field |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct aberrations, flat-field |
| **M2** Compound | Compound correction with rho measurement: Correct aberrations, flat-field |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct aberrations, flat-field |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct aberrations, flat-field |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
