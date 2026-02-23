# Dark-Field Microscopy (`dark_field`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: richardson_lucy

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Stop size, illumination angle, NA partitioning |
| **M1** Synthetic | Prompt tested with synthetic data validation: Stop size, illumination angle, NA partitioning |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Stop size, illumination angle, NA partitioning |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Stop size, illumination angle, NA partitioning |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Dark-Field Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Dark-field recon under illumination non-uniformity |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Dark-field recon under illumination non-uniformity |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Dark-field recon under illumination non-uniformity |
| **M3** Real Data | Real experimental data with measured mismatch: Dark-field recon under illumination non-uniformity |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Dark-field recon under illumination non-uniformity |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Condenser NA vs objective NA ratio | 1.2 | [1.0, 1.5] | - |
| Stray light | 0 | [0, 5%] | relative |
| Scattering angle range | correct | +/- 10% | - |

### Solvers & Expected Performance
- **Solver**: richardson_lucy

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate stop alignment, illumination asymmetry |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate stop alignment, illumination asymmetry |
| **M2** Compound | Compound parameter identification (3+ params): Estimate stop alignment, illumination asymmetry |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate stop alignment, illumination asymmetry |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate stop alignment, illumination asymmetry |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct stop positioning, flat-field correction |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct stop positioning, flat-field correction |
| **M2** Compound | Compound correction with rho measurement: Correct stop positioning, flat-field correction |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct stop positioning, flat-field correction |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct stop positioning, flat-field correction |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
