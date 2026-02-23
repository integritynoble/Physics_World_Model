# XFEL Serial Femtosecond Crystallography (SFX) (`xfel_sfx`)

**Category**: Ultrafast Imaging | **Canonical DAG**: M --> R --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: indexing_merge

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Pulse energy, jet speed, hit rate, detector geometry |
| **M1** Synthetic | Prompt tested with synthetic data validation: Pulse energy, jet speed, hit rate, detector geometry |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Pulse energy, jet speed, hit rate, detector geometry |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Pulse energy, jet speed, hit rate, detector geometry |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for XFEL Serial Femtosecond Crystallography (SFX) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Indexing + merging under geometry error, partiality |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Indexing + merging under geometry error, partiality |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Indexing + merging under geometry error, partiality |
| **M3** Real Data | Real experimental data with measured mismatch: Indexing + merging under geometry error, partiality |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Indexing + merging under geometry error, partiality |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Hit rate | 10% | [1%, 30%] | - |
| Indexing ambiguity | 0 | [0, 10%] of patterns | - |
| Partiality model error | 0 | [0, 20%] | - |
| Background from jet/carrier | 0 | [0, 30%] | - |

### Solvers & Expected Performance
- **Solver**: indexing_merge

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate detector geometry, beam center, partiality |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate detector geometry, beam center, partiality |
| **M2** Compound | Compound parameter identification (3+ params): Estimate detector geometry, beam center, partiality |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate detector geometry, beam center, partiality |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate detector geometry, beam center, partiality |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct geometry, partiality model, scaling |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct geometry, partiality model, scaling |
| **M2** Compound | Compound correction with rho measurement: Correct geometry, partiality model, scaling |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct geometry, partiality model, scaling |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct geometry, partiality model, scaling |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
