# Multispectral Satellite Imaging (`multispectral_sat`)

**Category**: Remote Sensing | **Canonical DAG**: M --> Sigma --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: pan_sharpening

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Band selection, spatial resolution, orbit |
| **M1** Synthetic | Prompt tested with synthetic data validation: Band selection, spatial resolution, orbit |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Band selection, spatial resolution, orbit |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Band selection, spatial resolution, orbit |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Multispectral Satellite Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Pan-sharpening under co-registration error, MTF difference |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Pan-sharpening under co-registration error, MTF difference |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Pan-sharpening under co-registration error, MTF difference |
| **M3** Real Data | Real experimental data with measured mismatch: Pan-sharpening under co-registration error, MTF difference |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Pan-sharpening under co-registration error, MTF difference |

### Mismatch Parameters
M→Sigma→D. Band registration [0,2] px, MTF +/-10%.

### Solvers & Expected Performance
- **Solver**: pan_sharpening

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> Sigma --> D: Estimate band-to-band registration, MTF per band |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate band-to-band registration, MTF per band |
| **M2** Compound | Compound parameter identification (3+ params): Estimate band-to-band registration, MTF per band |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate band-to-band registration, MTF per band |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate band-to-band registration, MTF per band |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct registration, MTF matching |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct registration, MTF matching |
| **M2** Compound | Compound correction with rho measurement: Correct registration, MTF matching |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct registration, MTF matching |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct registration, MTF matching |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
