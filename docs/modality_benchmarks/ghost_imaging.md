# Ghost Imaging (`ghost_imaging`)

**Category**: Quantum Imaging | **Canonical DAG**: M --> Sigma --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: correlation_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | "Design ghost imaging system: thermal source, DMD modulation, single-pixel bucket detector." |
| **M1** Synthetic | Prompt tested with synthetic data validation: Bucket detector, spatial patterns, photon rate |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Bucket detector, spatial patterns, photon rate |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Bucket detector, spatial patterns, photon rate |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Ghost Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Correlation reconstruction under accidental coincidences |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Correlation reconstruction under accidental coincidences |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Correlation reconstruction under accidental coincidences |
| **M3** Real Data | Real experimental data with measured mismatch: Correlation reconstruction under accidental coincidences |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Correlation reconstruction under accidental coincidences |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Bucket detector efficiency | 1.0 | [0.5, 1.0] | - |
| Speckle correlation mismatch | 0 | [0, 10%] | - |
| Background counts | 0 | [0, 5%] of signal | - |
| Number of measurements | 10000 | [1000, 100000] | - |

### Solvers & Expected Performance
- **Solver**: correlation_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> Sigma --> D: Estimate accidental rate, visibility, detection efficiency |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate accidental rate, visibility, detection efficiency |
| **M2** Compound | Compound parameter identification (3+ params): Estimate accidental rate, visibility, detection efficiency |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate accidental rate, visibility, detection efficiency |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate accidental rate, visibility, detection efficiency |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct accidentals, improve SNR via optimal patterns |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct accidentals, improve SNR via optimal patterns |
| **M2** Compound | Compound correction with rho measurement: Correct accidentals, improve SNR via optimal patterns |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct accidentals, improve SNR via optimal patterns |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct accidentals, improve SNR via optimal patterns |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Computational ghost imaging vs quantum ghost imaging comparison.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
