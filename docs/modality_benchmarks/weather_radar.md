# Weather / Doppler Radar (`weather_radar`)

**Category**: Remote Sensing | **Canonical DAG**: P --> R --> D | **Carrier**: RF
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: reflectivity_estimation

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Wavelength, scan strategy, PRF, dual-pol |
| **M1** Synthetic | Prompt tested with synthetic data validation: Wavelength, scan strategy, PRF, dual-pol |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Wavelength, scan strategy, PRF, dual-pol |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Wavelength, scan strategy, PRF, dual-pol |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Weather / Doppler Radar |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Reflectivity estimation under ground clutter, attenuation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Reflectivity estimation under ground clutter, attenuation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Reflectivity estimation under ground clutter, attenuation |
| **M3** Real Data | Real experimental data with measured mismatch: Reflectivity estimation under ground clutter, attenuation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Reflectivity estimation under ground clutter, attenuation |

### Mismatch Parameters
P→R→D, RF. Clutter [-40,-15] dBZ, attenuation [0,10] dB/km.

### Solvers & Expected Performance
- **Solver**: reflectivity_estimation

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> R --> D: Estimate clutter map, attenuation path |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate clutter map, attenuation path |
| **M2** Compound | Compound parameter identification (3+ params): Estimate clutter map, attenuation path |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate clutter map, attenuation path |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate clutter map, attenuation path |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct clutter filter, attenuation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct clutter filter, attenuation |
| **M2** Compound | Compound correction with rho measurement: Correct clutter filter, attenuation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct clutter filter, attenuation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct clutter filter, attenuation |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
