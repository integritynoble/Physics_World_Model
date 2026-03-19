# Contrast-Enhanced Ultrasound (CEUS) (`ceus`)

**Category**: Medical Imaging | **Canonical DAG**: P --> R --> D | **Carrier**: Acoustic
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: contrast_specific

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: MI, pulse scheme, frame rate, contrast agent |
| **M1** Synthetic | Prompt tested with synthetic data validation: MI, pulse scheme, frame rate, contrast agent |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for MI, pulse scheme, frame rate, contrast agent |
| **M3** Real Data | Grounded in real experimental/clinical protocols: MI, pulse scheme, frame rate, contrast agent |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Contrast-Enhanced Ultrasound (CEUS) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Contrast-specific imaging under tissue clutter |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Contrast-specific imaging under tissue clutter |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Contrast-specific imaging under tissue clutter |
| **M3** Real Data | Real experimental data with measured mismatch: Contrast-specific imaging under tissue clutter |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Contrast-specific imaging under tissue clutter |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Bubble concentration | optimal | [0.1x, 5x] | relative |
| Nonlinear harmonic extraction | clean | [0, 10%] tissue leak | - |
| Motion between frames | 0 | [0, 5] | mm |

### Solvers & Expected Performance
- **Solver**: contrast_specific

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> R --> D: Estimate tissue signal, bubble nonlinearity curve |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate tissue signal, bubble nonlinearity curve |
| **M2** Compound | Compound parameter identification (3+ params): Estimate tissue signal, bubble nonlinearity curve |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate tissue signal, bubble nonlinearity curve |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate tissue signal, bubble nonlinearity curve |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct tissue subtraction, linearize contrast |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct tissue subtraction, linearize contrast |
| **M2** Compound | Compound correction with rho measurement: Correct tissue subtraction, linearize contrast |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct tissue subtraction, linearize contrast |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct tissue subtraction, linearize contrast |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
