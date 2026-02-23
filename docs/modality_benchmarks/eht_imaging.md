# Event Horizon Telescope (EHT) Imaging (`eht_imaging`)

**Category**: Astronomy & Space Imaging | **Canonical DAG**: F --> S --> D | **Carrier**: RF (mm-wave)
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: rml_clean

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Baseline coverage, bandwidth, atmospheric opacity |
| **M1** Synthetic | Prompt tested with synthetic data validation: Baseline coverage, bandwidth, atmospheric opacity |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Baseline coverage, bandwidth, atmospheric opacity |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Baseline coverage, bandwidth, atmospheric opacity |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Event Horizon Telescope (EHT) Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: CLEAN/RML under atmospheric phase, sparse uv-coverage |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: CLEAN/RML under atmospheric phase, sparse uv-coverage |
| **M2** Compound | Compound mismatch (3+ params simultaneously): CLEAN/RML under atmospheric phase, sparse uv-coverage |
| **M3** Real Data | Real experimental data with measured mismatch: CLEAN/RML under atmospheric phase, sparse uv-coverage |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: CLEAN/RML under atmospheric phase, sparse uv-coverage |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Atmospheric opacity (tau) | 0.1 | [0.05, 0.5] | nepers |
| Station gain calibration | 0 | [0, 10%] per station | - |
| uv-coverage sparsity | sparse | varies by night | - |
| Interstellar scattering | 0 | [0, 10] | uas broadening |

### Solvers & Expected Performance
- **Solver**: rml_clean

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for F --> S --> D: Estimate station gains, atmospheric phase, bandpass |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate station gains, atmospheric phase, bandpass |
| **M2** Compound | Compound parameter identification (3+ params): Estimate station gains, atmospheric phase, bandpass |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate station gains, atmospheric phase, bandpass |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate station gains, atmospheric phase, bandpass |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct atmospheric phase, amplitude calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct atmospheric phase, amplitude calibration |
| **M2** Compound | Compound correction with rho measurement: Correct atmospheric phase, amplitude calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct atmospheric phase, amplitude calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct atmospheric phase, amplitude calibration |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Test different regularizers (MEM, RML, CLEAN, PRIMO).

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
