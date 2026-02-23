# Brachytherapy Imaging (`brachytherapy_img`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> D | **Carrier**: Gamma/X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: tg43_dose

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Source geometry, applicator model, imaging protocol |
| **M1** Synthetic | Prompt tested with synthetic data validation: Source geometry, applicator model, imaging protocol |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Source geometry, applicator model, imaging protocol |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Source geometry, applicator model, imaging protocol |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Brachytherapy Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: TG-43/TG-186 dose with imaging verification |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: TG-43/TG-186 dose with imaging verification |
| **M2** Compound | Compound mismatch (3+ params simultaneously): TG-43/TG-186 dose with imaging verification |
| **M3** Real Data | Real experimental data with measured mismatch: TG-43/TG-186 dose with imaging verification |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: TG-43/TG-186 dose with imaging verification |

### Mismatch Parameters
Pi→D, Gamma. Source position [0,3] mm, applicator [0,2] mm.

### Solvers & Expected Performance
- **Solver**: tg43_dose

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate source position, applicator geometry |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate source position, applicator geometry |
| **M2** Compound | Compound parameter identification (3+ params): Estimate source position, applicator geometry |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate source position, applicator geometry |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate source position, applicator geometry |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct source localization, applicator model |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct source localization, applicator model |
| **M2** Compound | Compound correction with rho measurement: Correct source localization, applicator model |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct source localization, applicator model |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct source localization, applicator model |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
