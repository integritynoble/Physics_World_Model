# Phase Contrast Microscopy (`phase_contrast`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: halo_removal

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Phase ring design, condenser annulus, NA match |
| **M1** Synthetic | Prompt tested with synthetic data validation: Phase ring design, condenser annulus, NA match |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Phase ring design, condenser annulus, NA match |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Phase ring design, condenser annulus, NA match |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Phase Contrast Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Halo artifact correction under ring misalignment |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Halo artifact correction under ring misalignment |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Halo artifact correction under ring misalignment |
| **M3** Real Data | Real experimental data with measured mismatch: Halo artifact correction under ring misalignment |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Halo artifact correction under ring misalignment |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Phase ring alignment | 0 | [0, 5] | um offset |
| Halo artifact strength | 0 | [0, 0.3] | relative |
| Phase ring absorption | 0.7 | [0.5, 0.9] | - |

### Solvers & Expected Performance
- **Solver**: halo_removal

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate phase ring position, absorption ratio |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate phase ring position, absorption ratio |
| **M2** Compound | Compound parameter identification (3+ params): Estimate phase ring position, absorption ratio |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate phase ring position, absorption ratio |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate phase ring position, absorption ratio |

### True-Spec Parameters
Phase ring position, absorption coefficient, condenser alignment

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct phase ring alignment, halo suppression |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct phase ring alignment, halo suppression |
| **M2** Compound | Compound correction with rho measurement: Correct phase ring alignment, halo suppression |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct phase ring alignment, halo suppression |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct phase ring alignment, halo suppression |

### Correction Targets
- **Expected rho**: >= 0.70

### Improvement Roadmap
Test quantitative phase recovery from phase contrast images.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
