# Structured Illumination Microscopy (SIM) (`sim`)

**Category**: Microscopy | **Canonical DAG**: M --> C --> D | **Carrier**: Photon
**Current Maturity**: M2 | **Forward Model**: linear_operator | **Default Solver**: wiener_sim

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Pattern frequency, orientations, phase steps |
| **M1** Synthetic | Prompt tested with synthetic data validation: Pattern frequency, orientations, phase steps |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Pattern frequency, orientations, phase steps |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Pattern frequency, orientations, phase steps |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Structured Illumination Microscopy (SIM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Wiener-SIM reconstruction under pattern mismatch |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Wiener-SIM reconstruction under pattern mismatch |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Wiener-SIM reconstruction under pattern mismatch |
| **M3** Real Data | Real experimental data with measured mismatch: Wiener-SIM reconstruction under pattern mismatch |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Wiener-SIM reconstruction under pattern mismatch |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Pattern frequency | 0.1 | [0.05, 0.15] | cycles/px |
| Phase shifts | [0, 2pi/3, 4pi/3] | +/- 0.2 rad each | rad |
| Modulation depth | 0.8 | [0.3, 1.0] | - |
| Pattern orientation | [0, 60, 120] | +/- 3 deg each | deg |

### Solvers & Expected Performance
- **Solver(s)**: Wiener-SIM, HiFi-SIM, fairSIM
- **Scenario I PSNR**: 28-35 dB
- **Scenario II drop**: 5-12 dB

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> C --> D: Estimate pattern freq, phase, modulation depth |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate pattern freq, phase, modulation depth |
| **M2** Compound | Compound parameter identification (3+ params): Estimate pattern freq, phase, modulation depth |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate pattern freq, phase, modulation depth |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate pattern freq, phase, modulation depth |

### True-Spec Parameters
Frequencies (3), phases (9), modulation depths (3), orientations (3), OTF

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct illumination pattern errors |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct illumination pattern errors |
| **M2** Compound | Compound correction with rho measurement: Correct illumination pattern errors |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct illumination pattern errors |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct illumination pattern errors |

### Correction Targets
- **Expected rho**: >= 0.80

### Improvement Roadmap
Add 3D-SIM, nonlinear SIM, compound mismatch.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
