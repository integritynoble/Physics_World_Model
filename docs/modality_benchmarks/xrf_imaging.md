# X-ray Fluorescence (XRF) Imaging (`xrf_imaging`)

**Category**: Industrial Inspection | **Canonical DAG**: M --> R --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: element_quantification

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Excitation energy, detector geometry, resolution |
| **M1** Synthetic | Prompt tested with synthetic data validation: Excitation energy, detector geometry, resolution |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Excitation energy, detector geometry, resolution |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Excitation energy, detector geometry, resolution |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for X-ray Fluorescence (XRF) Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Element mapping under matrix effect, self-absorption |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Element mapping under matrix effect, self-absorption |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Element mapping under matrix effect, self-absorption |
| **M3** Real Data | Real experimental data with measured mismatch: Element mapping under matrix effect, self-absorption |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Element mapping under matrix effect, self-absorption |

### Mismatch Parameters
M→R→D. Matrix effects [0,20%], self-absorption [0,30%], dead time [0,10%].

### Solvers & Expected Performance
- **Solver**: element_quantification

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate matrix composition, self-absorption |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate matrix composition, self-absorption |
| **M2** Compound | Compound parameter identification (3+ params): Estimate matrix composition, self-absorption |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate matrix composition, self-absorption |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate matrix composition, self-absorption |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct matrix effects, dead time, pile-up |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct matrix effects, dead time, pile-up |
| **M2** Compound | Compound correction with rho measurement: Correct matrix effects, dead time, pile-up |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct matrix effects, dead time, pile-up |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct matrix effects, dead time, pile-up |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
