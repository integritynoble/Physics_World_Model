# Wide-Angle X-ray Scattering (WAXS) (`waxs`)

**Category**: Scientific Instrumentation | **Canonical DAG**: R --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: azimuthal_integration

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Detector geometry, q-range, polarization correction |
| **M1** Synthetic | Prompt tested with synthetic data validation: Detector geometry, q-range, polarization correction |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Detector geometry, q-range, polarization correction |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Detector geometry, q-range, polarization correction |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Wide-Angle X-ray Scattering (WAXS) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Azimuthal integration under detector tilt, beam center |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Azimuthal integration under detector tilt, beam center |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Azimuthal integration under detector tilt, beam center |
| **M3** Real Data | Real experimental data with measured mismatch: Azimuthal integration under detector tilt, beam center |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Azimuthal integration under detector tilt, beam center |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Detector distance error | 0 | [-1%, 1%] | - |
| Beam center error | 0 | [0, 3] | px |
| Polarization correction | 1.0 | [0.9, 1.0] | - |
| Air scatter background | 0 | [0, 5%] | - |

### Solvers & Expected Performance
- **Solver**: azimuthal_integration

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for R --> D: Estimate beam center, tilt angles, flat-field |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate beam center, tilt angles, flat-field |
| **M2** Compound | Compound parameter identification (3+ params): Estimate beam center, tilt angles, flat-field |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate beam center, tilt angles, flat-field |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate beam center, tilt angles, flat-field |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct detector geometry, polarization |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct detector geometry, polarization |
| **M2** Compound | Compound correction with rho measurement: Correct detector geometry, polarization |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct detector geometry, polarization |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct detector geometry, polarization |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
