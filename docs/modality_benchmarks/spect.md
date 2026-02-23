# Single Photon Emission CT (SPECT) (`spect`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> D | **Carrier**: Gamma
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: mlem

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Collimator type, orbit, energy window, attenuation correction |
| **M1** Synthetic | Prompt tested with synthetic data validation: Collimator type, orbit, energy window, attenuation correction |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Collimator type, orbit, energy window, attenuation correction |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Collimator type, orbit, energy window, attenuation correction |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Single Photon Emission CT (SPECT) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: MLEM with depth-dependent resolution under collimator response error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: MLEM with depth-dependent resolution under collimator response error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): MLEM with depth-dependent resolution under collimator response error |
| **M3** Real Data | Real experimental data with measured mismatch: MLEM with depth-dependent resolution under collimator response error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: MLEM with depth-dependent resolution under collimator response error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Collimator response error | 0 | [0, 20%] FWHM | - |
| Center-of-rotation | 0 | [-3, 3] | px |
| Attenuation error | 0 | [0, 15%] | relative |

### Solvers & Expected Performance
- **Solver**: mlem

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate collimator params, attenuation map, center-of-rotation |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate collimator params, attenuation map, center-of-rotation |
| **M2** Compound | Compound parameter identification (3+ params): Estimate collimator params, attenuation map, center-of-rotation |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate collimator params, attenuation map, center-of-rotation |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate collimator params, attenuation map, center-of-rotation |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct CoR, collimator model, attenuation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct CoR, collimator model, attenuation |
| **M2** Compound | Compound correction with rho measurement: Correct CoR, collimator model, attenuation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct CoR, collimator model, attenuation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct CoR, collimator model, attenuation |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
