# DNA-PAINT Super-Resolution (`dna_paint`)

**Category**: Microscopy | **Canonical DAG**: M --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: thunderstorm

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Imager strand concentration, binding kinetics |
| **M1** Synthetic | Prompt tested with synthetic data validation: Imager strand concentration, binding kinetics |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Imager strand concentration, binding kinetics |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Imager strand concentration, binding kinetics |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for DNA-PAINT Super-Resolution |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Localization under binding rate variation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Localization under binding rate variation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Localization under binding rate variation |
| **M3** Real Data | Real experimental data with measured mismatch: Localization under binding rate variation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Localization under binding rate variation |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Binding on-rate | varies | [0.5x, 2.0x] | relative |
| Imager strand concentration | 5 | [1, 20] | nM |
| Drift rate | 0 | [0, 3] | nm/frame |
| Background from non-specific binding | 0 | [0, 10%] | - |

### Solvers & Expected Performance
- **Solver**: thunderstorm

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> D: Estimate on-rate, off-rate, background binding |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate on-rate, off-rate, background binding |
| **M2** Compound | Compound parameter identification (3+ params): Estimate on-rate, off-rate, background binding |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate on-rate, off-rate, background binding |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate on-rate, off-rate, background binding |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct binding kinetics model, background |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct binding kinetics model, background |
| **M2** Compound | Compound correction with rho measurement: Correct binding kinetics model, background |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct binding kinetics model, background |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct binding kinetics model, background |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Add Exchange-PAINT (multi-target) benchmark.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
