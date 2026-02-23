# Laser-Induced Breakdown Spectroscopy (LIBS) Imaging (`libs`)

**Category**: Spectroscopy & Spectral Imaging | **Canonical DAG**: M --> R --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: element_quantification

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Laser energy, gate delay, spot size, spectrometer |
| **M1** Synthetic | Prompt tested with synthetic data validation: Laser energy, gate delay, spot size, spectrometer |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Laser energy, gate delay, spot size, spectrometer |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Laser energy, gate delay, spot size, spectrometer |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Laser-Induced Breakdown Spectroscopy (LIBS) Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Element quantification under matrix effect, self-absorption |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Element quantification under matrix effect, self-absorption |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Element quantification under matrix effect, self-absorption |
| **M3** Real Data | Real experimental data with measured mismatch: Element quantification under matrix effect, self-absorption |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Element quantification under matrix effect, self-absorption |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Laser energy fluctuation | 0 | [0, 10%] | - |
| Matrix effect | 0 | [0, 30%] | - |
| Self-absorption correction | 0 | [0, 20%] | - |
| Crater-to-crater variation | 0 | [0, 15%] | - |

### Solvers & Expected Performance
- **Solver**: element_quantification

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate plasma temperature, electron density |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate plasma temperature, electron density |
| **M2** Compound | Compound parameter identification (3+ params): Estimate plasma temperature, electron density |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate plasma temperature, electron density |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate plasma temperature, electron density |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct matrix effects, self-absorption |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct matrix effects, self-absorption |
| **M2** Compound | Compound correction with rho measurement: Correct matrix effects, self-absorption |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct matrix effects, self-absorption |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct matrix effects, self-absorption |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
