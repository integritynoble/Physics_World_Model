# Coded Exposure / Flutter Shutter (`coded_exposure`)

**Category**: Computational Photography | **Canonical DAG**: M --> C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: wiener_deblur

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Shutter code, exposure time, motion range |
| **M1** Synthetic | Prompt tested with synthetic data validation: Shutter code, exposure time, motion range |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Shutter code, exposure time, motion range |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Shutter code, exposure time, motion range |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Coded Exposure / Flutter Shutter |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Deblurring under code timing error, unknown motion |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Deblurring under code timing error, unknown motion |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Deblurring under code timing error, unknown motion |
| **M3** Real Data | Real experimental data with measured mismatch: Deblurring under code timing error, unknown motion |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Deblurring under code timing error, unknown motion |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Shutter code timing error | 0 | [-5%, 5%] per slot | - |
| Motion blur PSF mismatch | 0 | [0, 20%] | velocity error |
| Sensor readout noise | 5 | [1, 15] | e- |

### Solvers & Expected Performance
- **Solver**: wiener_deblur

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> C --> D: Estimate shutter function, motion kernel |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate shutter function, motion kernel |
| **M2** Compound | Compound parameter identification (3+ params): Estimate shutter function, motion kernel |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate shutter function, motion kernel |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate shutter function, motion kernel |

### True-Spec Parameters
Exact shutter timing sequence, true motion velocity

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct shutter timing, motion model |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct shutter timing, motion model |
| **M2** Compound | Compound correction with rho measurement: Correct shutter timing, motion model |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct shutter timing, motion model |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct shutter timing, motion model |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
