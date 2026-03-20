# Coherent Diffractive Imaging / Phase Retrieval (`phase_retrieval`)

**Category**: Coherent Imaging | **Canonical DAG**: P --> D | **Carrier**: Photon/Electron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: hio

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Support constraint, oversampling ratio, coherence |
| **M1** Synthetic | Prompt tested with synthetic data validation: Support constraint, oversampling ratio, coherence |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Support constraint, oversampling ratio, coherence |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Support constraint, oversampling ratio, coherence |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Coherent Diffractive Imaging / Phase Retrieval |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: HIO/ER under support error, partial coherence |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: HIO/ER under support error, partial coherence |
| **M2** Compound | Compound mismatch (3+ params simultaneously): HIO/ER under support error, partial coherence |
| **M3** Real Data | Real experimental data with measured mismatch: HIO/ER under support error, partial coherence |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: HIO/ER under support error, partial coherence |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Support mask error | 0 | [0, 10%] area | - |
| Oversampling ratio | 2.0 | [1.5, 4.0] | - |
| Partial coherence | 1.0 | [0.7, 1.0] | - |

### Solvers & Expected Performance
- **Solver**: hio

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate support boundary, coherence function |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate support boundary, coherence function |
| **M2** Compound | Compound parameter identification (3+ params): Estimate support boundary, coherence function |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate support boundary, coherence function |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate support boundary, coherence function |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct support, coherence model |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct support, coherence model |
| **M2** Compound | Compound correction with rho measurement: Correct support, coherence model |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct support, coherence model |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct support, coherence model |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
