# Photoacoustic Imaging (`photoacoustic`)

**Category**: Medical Imaging | **Canonical DAG**: M --> P --> D | **Carrier**: Acoustic
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: back_projection

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Transducer array, laser wavelength, fluence model |
| **M1** Synthetic | Prompt tested with synthetic data validation: Transducer array, laser wavelength, fluence model |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Transducer array, laser wavelength, fluence model |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Transducer array, laser wavelength, fluence model |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Photoacoustic Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Backprojection under speed-of-sound heterogeneity, acoustic attenuation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Backprojection under speed-of-sound heterogeneity, acoustic attenuation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Backprojection under speed-of-sound heterogeneity, acoustic attenuation |
| **M3** Real Data | Real experimental data with measured mismatch: Backprojection under speed-of-sound heterogeneity, acoustic attenuation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Backprojection under speed-of-sound heterogeneity, acoustic attenuation |

### Mismatch Parameters
M → P → D, Acoustic. Speed [1400,1600] m/s, attenuation [0,0.5] dB/cm/MHz, fluence [0,30%].

### Solvers & Expected Performance
- **Solver**: back_projection

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> P --> D: Estimate sound speed map, Grueneisen parameter, fluence |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate sound speed map, Grueneisen parameter, fluence |
| **M2** Compound | Compound parameter identification (3+ params): Estimate sound speed map, Grueneisen parameter, fluence |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate sound speed map, Grueneisen parameter, fluence |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate sound speed map, Grueneisen parameter, fluence |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct sound speed model, fluence compensation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct sound speed model, fluence compensation |
| **M2** Compound | Compound correction with rho measurement: Correct sound speed model, fluence compensation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct sound speed model, fluence compensation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct sound speed model, fluence compensation |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
