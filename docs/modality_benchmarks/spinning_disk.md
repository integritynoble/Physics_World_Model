# Spinning Disk Confocal Microscopy (`spinning_disk`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: richardson_lucy

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Pinhole spacing, rotation speed, camera sync |
| **M1** Synthetic | Prompt tested with synthetic data validation: Pinhole spacing, rotation speed, camera sync |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Pinhole spacing, rotation speed, camera sync |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Pinhole spacing, rotation speed, camera sync |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Spinning Disk Confocal Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Deconvolution under pinhole crosstalk |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Deconvolution under pinhole crosstalk |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Deconvolution under pinhole crosstalk |
| **M3** Real Data | Real experimental data with measured mismatch: Deconvolution under pinhole crosstalk |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Deconvolution under pinhole crosstalk |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Pinhole crosstalk | 0 | [0, 15%] | - |
| Disk rotation wobble | 0 | [0, 1] | px |
| Illumination non-uniformity | 0 | [0, 10%] | - |

### Solvers & Expected Performance
- **Solver**: richardson_lucy

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate pinhole spacing error, crosstalk, sync lag |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate pinhole spacing error, crosstalk, sync lag |
| **M2** Compound | Compound parameter identification (3+ params): Estimate pinhole spacing error, crosstalk, sync lag |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate pinhole spacing error, crosstalk, sync lag |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate pinhole spacing error, crosstalk, sync lag |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct crosstalk, synchronization timing |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct crosstalk, synchronization timing |
| **M2** Compound | Compound correction with rho measurement: Correct crosstalk, synchronization timing |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct crosstalk, synchronization timing |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct crosstalk, synchronization timing |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
