# Solar EUV/X-ray Imaging (`solar_imaging`)

**Category**: Astronomy & Space Imaging | **Canonical DAG**: M --> P --> D | **Carrier**: Photon/EUV
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: dem_inversion

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Wavelength channels, cadence, pointing stability |
| **M1** Synthetic | Prompt tested with synthetic data validation: Wavelength channels, cadence, pointing stability |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Wavelength channels, cadence, pointing stability |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Wavelength channels, cadence, pointing stability |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Solar EUV/X-ray Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: DEM reconstruction under PSF degradation, stray light |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: DEM reconstruction under PSF degradation, stray light |
| **M2** Compound | Compound mismatch (3+ params simultaneously): DEM reconstruction under PSF degradation, stray light |
| **M3** Real Data | Real experimental data with measured mismatch: DEM reconstruction under PSF degradation, stray light |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: DEM reconstruction under PSF degradation, stray light |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| PSF degradation (mirror aging) | 0 | [0, 20%] | - |
| Stray light | 0 | [0, 5%] | - |
| Flat-field error | 0 | [0, 3%] | - |
| Pointing jitter | 0 | [0, 1] | arcsec |

### Solvers & Expected Performance
- **Solver**: dem_inversion

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> P --> D: Estimate PSF degradation curve, stray light model |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate PSF degradation curve, stray light model |
| **M2** Compound | Compound parameter identification (3+ params): Estimate PSF degradation curve, stray light model |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate PSF degradation curve, stray light model |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate PSF degradation curve, stray light model |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct PSF evolution, stray light subtraction |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct PSF evolution, stray light subtraction |
| **M2** Compound | Compound correction with rho measurement: Correct PSF evolution, stray light subtraction |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct PSF evolution, stray light subtraction |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct PSF evolution, stray light subtraction |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
