# Quantum Illumination (`quantum_illumination`)

**Category**: Quantum Imaging | **Canonical DAG**: M --> R --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: quantum_detector

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Signal-idler source, entanglement quality, detector |
| **M1** Synthetic | Prompt tested with synthetic data validation: Signal-idler source, entanglement quality, detector |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Signal-idler source, entanglement quality, detector |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Signal-idler source, entanglement quality, detector |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Quantum Illumination |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Target detection under background thermal noise |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Target detection under background thermal noise |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Target detection under background thermal noise |
| **M3** Real Data | Real experimental data with measured mismatch: Target detection under background thermal noise |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Target detection under background thermal noise |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Entanglement quality (concurrence) | 1.0 | [0.5, 1.0] | - |
| Background thermal noise | 0 | [0, 100] photons/mode | - |
| Detector dark count rate | 0 | [0, 1000] | Hz |
| Channel loss | 0 | [0, 30] | dB |

### Solvers & Expected Performance
- **Solver**: quantum_detector

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate entanglement quality, background level |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate entanglement quality, background level |
| **M2** Compound | Compound parameter identification (3+ params): Estimate entanglement quality, background level |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate entanglement quality, background level |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate entanglement quality, background level |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct noise model, optimize detection threshold |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct noise model, optimize detection threshold |
| **M2** Compound | Compound correction with rho measurement: Correct noise model, optimize detection threshold |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct noise model, optimize detection threshold |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct noise model, optimize detection threshold |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
