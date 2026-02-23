# Ocean Color Remote Sensing (`ocean_color`)

**Category**: Remote Sensing | **Canonical DAG**: M --> Sigma --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: atmospheric_correction

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Spectral bands, spatial resolution, sun glint avoidance |
| **M1** Synthetic | Prompt tested with synthetic data validation: Spectral bands, spatial resolution, sun glint avoidance |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Spectral bands, spatial resolution, sun glint avoidance |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Spectral bands, spatial resolution, sun glint avoidance |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Ocean Color Remote Sensing |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Atmospheric correction under aerosol model error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Atmospheric correction under aerosol model error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Atmospheric correction under aerosol model error |
| **M3** Real Data | Real experimental data with measured mismatch: Atmospheric correction under aerosol model error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Atmospheric correction under aerosol model error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Atmospheric correction error | 0 | [0, 15%] | - |
| Sun glint contamination | 0 | [0, 20%] of pixels | - |
| Vicarious calibration offset | 0 | [-3%, 3%] per band | - |

### Solvers & Expected Performance
- **Solver**: atmospheric_correction

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> Sigma --> D: Estimate aerosol optical depth, water-leaving radiance |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate aerosol optical depth, water-leaving radiance |
| **M2** Compound | Compound parameter identification (3+ params): Estimate aerosol optical depth, water-leaving radiance |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate aerosol optical depth, water-leaving radiance |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate aerosol optical depth, water-leaving radiance |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct atmospheric path, sun glint removal |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct atmospheric path, sun glint removal |
| **M2** Compound | Compound correction with rho measurement: Correct atmospheric path, sun glint removal |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct atmospheric path, sun glint removal |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct atmospheric path, sun glint removal |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
