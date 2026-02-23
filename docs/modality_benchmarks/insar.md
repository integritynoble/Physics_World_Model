# Interferometric SAR (InSAR) (`insar`)

**Category**: Remote Sensing | **Canonical DAG**: F --> S --> D | **Carrier**: RF
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: phase_unwrap

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Baseline, temporal separation, coherence |
| **M1** Synthetic | Prompt tested with synthetic data validation: Baseline, temporal separation, coherence |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Baseline, temporal separation, coherence |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Baseline, temporal separation, coherence |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Interferometric SAR (InSAR) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Phase unwrapping under atmospheric delay, decorrelation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Phase unwrapping under atmospheric delay, decorrelation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Phase unwrapping under atmospheric delay, decorrelation |
| **M3** Real Data | Real experimental data with measured mismatch: Phase unwrapping under atmospheric delay, decorrelation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Phase unwrapping under atmospheric delay, decorrelation |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Phase unwrapping error | 0 | [0, 5%] of pixels | - |
| Baseline estimation error | 0 | [0, 1] | m |
| Atmospheric phase screen | 0 | [0, 1] | rad rms |
| Temporal decorrelation | 0 | [0, 0.3] | coherence loss |

### Solvers & Expected Performance
- **Solver**: phase_unwrap

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for F --> S --> D: Estimate atmospheric phase screen, coherence map |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate atmospheric phase screen, coherence map |
| **M2** Compound | Compound parameter identification (3+ params): Estimate atmospheric phase screen, coherence map |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate atmospheric phase screen, coherence map |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate atmospheric phase screen, coherence map |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct atmospheric delay, improve coherence |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct atmospheric delay, improve coherence |
| **M2** Compound | Compound correction with rho measurement: Correct atmospheric delay, improve coherence |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct atmospheric delay, improve coherence |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct atmospheric delay, improve coherence |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
DInSAR for deformation, time-series InSAR (SBAS, PSI).

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
