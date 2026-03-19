# Ultrasound B-mode Imaging (`ultrasound`)

**Category**: Medical Imaging | **Canonical DAG**: P --> D | **Carrier**: Acoustic
**Current Maturity**: M1 | **Forward Model**: linear_operator | **Default Solver**: das_beamform

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Transducer array, frequency, focus depth, apodization |
| **M1** Synthetic | Prompt tested with synthetic data validation: Transducer array, frequency, focus depth, apodization |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Transducer array, frequency, focus depth, apodization |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Transducer array, frequency, focus depth, apodization |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Ultrasound B-mode Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: DAS beamforming under speed-of-sound error, phase aberration |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: DAS beamforming under speed-of-sound error, phase aberration |
| **M2** Compound | Compound mismatch (3+ params simultaneously): DAS beamforming under speed-of-sound error, phase aberration |
| **M3** Real Data | Real experimental data with measured mismatch: DAS beamforming under speed-of-sound error, phase aberration |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: DAS beamforming under speed-of-sound error, phase aberration |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Speed of sound | 1540 | [1450, 1600] | m/s |
| Phase aberration | 0 | [0, 50] | ns rms |
| Element sensitivity | 1.0 | [0.7, 1.3] per elem | - |
| Attenuation | 0.5 | [0.3, 0.8] | dB/cm/MHz |

### Solvers & Expected Performance
- **Solver**: das_beamform

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate sound speed profile, aberration screen |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate sound speed profile, aberration screen |
| **M2** Compound | Compound parameter identification (3+ params): Estimate sound speed profile, aberration screen |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate sound speed profile, aberration screen |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate sound speed profile, aberration screen |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct aberration, adaptive beamforming |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct aberration, adaptive beamforming |
| **M2** Compound | Compound correction with rho measurement: Correct aberration, adaptive beamforming |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct aberration, adaptive beamforming |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct aberration, adaptive beamforming |

### Correction Targets
- **Expected rho**: >= 0.70

### Improvement Roadmap
Aberration correction, plane-wave ultrafast.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
