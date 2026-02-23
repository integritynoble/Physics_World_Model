# Radio Aperture Synthesis (`radio_astronomy`)

**Category**: Broader Experimental Science | **Canonical DAG**: F --> S --> D | **Carrier**: RF
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: clean

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Antenna configuration, bandwidth, uv-coverage |
| **M1** Synthetic | Prompt tested with synthetic data validation: Antenna configuration, bandwidth, uv-coverage |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Antenna configuration, bandwidth, uv-coverage |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Antenna configuration, bandwidth, uv-coverage |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Radio Aperture Synthesis |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: CLEAN / MEM under baseline phase error, RFI |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: CLEAN / MEM under baseline phase error, RFI |
| **M2** Compound | Compound mismatch (3+ params simultaneously): CLEAN / MEM under baseline phase error, RFI |
| **M3** Real Data | Real experimental data with measured mismatch: CLEAN / MEM under baseline phase error, RFI |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: CLEAN / MEM under baseline phase error, RFI |

### Mismatch Parameters
F→S→D. Antenna gain [0,5%], phase [0,10] deg, RFI [0,5%].

### Solvers & Expected Performance
- **Solver**: clean

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for F --> S --> D: Estimate antenna gains, phase offsets, RFI |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate antenna gains, phase offsets, RFI |
| **M2** Compound | Compound parameter identification (3+ params): Estimate antenna gains, phase offsets, RFI |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate antenna gains, phase offsets, RFI |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate antenna gains, phase offsets, RFI |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct antenna calibration, RFI excision |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct antenna calibration, RFI excision |
| **M2** Compound | Compound correction with rho measurement: Correct antenna calibration, RFI excision |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct antenna calibration, RFI excision |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct antenna calibration, RFI excision |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
