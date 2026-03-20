# Talbot-Lau X-ray Grating Interferometry (`talbot_lau`)

**Category**: Coherent Imaging | **Canonical DAG**: M --> P --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: phase_stepping

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Grating periods, design energy, inter-grating distance |
| **M1** Synthetic | Prompt tested with synthetic data validation: Grating periods, design energy, inter-grating distance |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Grating periods, design energy, inter-grating distance |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Grating periods, design energy, inter-grating distance |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Talbot-Lau X-ray Grating Interferometry |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Phase stepping under grating misalignment, vibration |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Phase stepping under grating misalignment, vibration |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Phase stepping under grating misalignment, vibration |
| **M3** Real Data | Real experimental data with measured mismatch: Phase stepping under grating misalignment, vibration |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Phase stepping under grating misalignment, vibration |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Grating alignment (rotation) | 0 | [-0.5, 0.5] | deg |
| Inter-grating distance error | 0 | [-1%, 1%] | - |
| Phase stepping error | 0 | [-5%, 5%] | per step |
| Grating defect fraction | 0 | [0, 3%] | - |

### Solvers & Expected Performance
- **Solver**: phase_stepping

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> P --> D: Estimate grating positions, period mismatch, visibility |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate grating positions, period mismatch, visibility |
| **M2** Compound | Compound parameter identification (3+ params): Estimate grating positions, period mismatch, visibility |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate grating positions, period mismatch, visibility |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate grating positions, period mismatch, visibility |

### True-Spec Parameters
Grating periods, distances, phase steps, defect map

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct grating alignment, period matching |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct grating alignment, period matching |
| **M2** Compound | Compound correction with rho measurement: Correct grating alignment, period matching |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct grating alignment, period matching |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct grating alignment, period matching |

### Correction Targets
- **Expected rho**: >= 0.75

### Improvement Roadmap
Simultaneous absorption/phase/dark-field reconstruction.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
