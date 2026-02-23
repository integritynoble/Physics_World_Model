# Event Camera / Dynamic Vision Sensor (DVS) (`event_camera`)

**Category**: Computational Photography | **Canonical DAG**: M --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: event_to_frame

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Contrast threshold, temporal resolution, bias settings |
| **M1** Synthetic | Prompt tested with synthetic data validation: Contrast threshold, temporal resolution, bias settings |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Contrast threshold, temporal resolution, bias settings |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Contrast threshold, temporal resolution, bias settings |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Event Camera / Dynamic Vision Sensor (DVS) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Event-to-frame under threshold mismatch, refractory period |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Event-to-frame under threshold mismatch, refractory period |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Event-to-frame under threshold mismatch, refractory period |
| **M3** Real Data | Real experimental data with measured mismatch: Event-to-frame under threshold mismatch, refractory period |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Event-to-frame under threshold mismatch, refractory period |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Contrast threshold | 0.3 | [0.1, 0.5] | log intensity |
| Refractory period | 1 | [0.1, 10] | us |
| Noise event rate | 0 | [0, 1%] | of real events |
| Hot pixel fraction | 0 | [0, 0.5%] | - |

### Solvers & Expected Performance
- **Solver**: event_to_frame

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> D: Estimate threshold per pixel, refractory time |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate threshold per pixel, refractory time |
| **M2** Compound | Compound parameter identification (3+ params): Estimate threshold per pixel, refractory time |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate threshold per pixel, refractory time |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate threshold per pixel, refractory time |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct threshold calibration, hot pixels |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct threshold calibration, hot pixels |
| **M2** Compound | Compound correction with rho measurement: Correct threshold calibration, hot pixels |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct threshold calibration, hot pixels |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct threshold calibration, hot pixels |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Test HDR reconstruction, high-speed video reconstruction from events.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
