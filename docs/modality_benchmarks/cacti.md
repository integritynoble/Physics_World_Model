# Coded Aperture Compressive Temporal Imaging (CACTI) (`cacti`)

**Category**: Compressive Imaging | **Canonical DAG**: M --> Sigma --> D | **Carrier**: Photon
**Current Maturity**: M3 | **Forward Model**: linear_operator | **Default Solver**: gap_tv

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Mask shift type, compression ratio, temporal resolution |
| **M1** Synthetic | Prompt tested with synthetic data validation: Mask shift type, compression ratio, temporal resolution |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Mask shift type, compression ratio, temporal resolution |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Mask shift type, compression ratio, temporal resolution |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Coded Aperture Compressive Temporal Imaging (CACTI) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: GAP-TV under spatial shift, rotation, temporal clock, gain, offset |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: GAP-TV under spatial shift, rotation, temporal clock, gain, offset |
| **M2** Compound | Compound mismatch (3+ params simultaneously): GAP-TV under spatial shift, rotation, temporal clock, gain, offset |
| **M3** Real Data | Real experimental data with measured mismatch: GAP-TV under spatial shift, rotation, temporal clock, gain, offset |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: GAP-TV under spatial shift, rotation, temporal clock, gain, offset |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Spatial shift x,y | 0 | [-3, 3] | px |
| Rotation | 0 | [-2, 2] | deg |
| Temporal clock error | 0 | [-0.5, 0.5] | frame frac |
| Gain / offset | 1.0 / 0 | [0.9,1.1] / [-5,5] | - / counts |
| Frame-dependent gain | 1.0 | [0.9, 1.1] per frame | - |

### Solvers & Expected Performance
- **Solver**: gap_tv
- **Validated baseline**: GAP-TV +10.21 dB, rho = 1.00

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> Sigma --> D: Estimate 8 mismatch params from temporal correlations |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate 8 mismatch params from temporal correlations |
| **M2** Compound | Compound parameter identification (3+ params): Estimate 8 mismatch params from temporal correlations |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate 8 mismatch params from temporal correlations |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate 8 mismatch params from temporal correlations |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct mask replication errors; rho validated at 100% |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct mask replication errors; rho validated at 100% |
| **M2** Compound | Compound correction with rho measurement: Correct mask replication errors; rho validated at 100% |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct mask replication errors; rho validated at 100% |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct mask replication errors; rho validated at 100% |

### Correction Targets
- **Expected rho**: 1.00

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
