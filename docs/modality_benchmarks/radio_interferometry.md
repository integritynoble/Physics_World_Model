# Radio Interferometry (VLBI) (`radio_interferometry`)

**Category**: Remote Sensing | **Canonical DAG**: F --> S --> D | **Carrier**: RF
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: clean

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Baseline configuration, bandwidth, integration time |
| **M1** Synthetic | Prompt tested with synthetic data validation: Baseline configuration, bandwidth, integration time |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Baseline configuration, bandwidth, integration time |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Baseline configuration, bandwidth, integration time |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Radio Interferometry (VLBI) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: CLEAN / MEM under baseline error, atmospheric phase |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: CLEAN / MEM under baseline error, atmospheric phase |
| **M2** Compound | Compound mismatch (3+ params simultaneously): CLEAN / MEM under baseline error, atmospheric phase |
| **M3** Real Data | Real experimental data with measured mismatch: CLEAN / MEM under baseline error, atmospheric phase |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: CLEAN / MEM under baseline error, atmospheric phase |

### Mismatch Parameters
F→S→D, RF. Baseline [0,1] cm, atmospheric phase [0,1] rad.

### Solvers & Expected Performance
- **Solver**: clean

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for F --> S --> D: Estimate baseline errors, atmospheric phase |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate baseline errors, atmospheric phase |
| **M2** Compound | Compound parameter identification (3+ params): Estimate baseline errors, atmospheric phase |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate baseline errors, atmospheric phase |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate baseline errors, atmospheric phase |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct baseline, atmospheric phase |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct baseline, atmospheric phase |
| **M2** Compound | Compound correction with rho measurement: Correct baseline, atmospheric phase |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct baseline, atmospheric phase |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct baseline, atmospheric phase |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
