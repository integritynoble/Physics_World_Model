# Mammography (`mammography`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: tv_fista

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Target/filter combination, compression, detector type |
| **M1** Synthetic | Prompt tested with synthetic data validation: Target/filter combination, compression, detector type |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Target/filter combination, compression, detector type |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Target/filter combination, compression, detector type |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Mammography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: TV-FISTA under scatter, heel effect, detector MTF variation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: TV-FISTA under scatter, heel effect, detector MTF variation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): TV-FISTA under scatter, heel effect, detector MTF variation |
| **M3** Real Data | Real experimental data with measured mismatch: TV-FISTA under scatter, heel effect, detector MTF variation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: TV-FISTA under scatter, heel effect, detector MTF variation |

### Mismatch Parameters
Pi → D, X-ray. Heel effect +/-10%, scatter-to-primary [0.2,0.8], MTF [0.8,1.2].

### Solvers & Expected Performance
- **Solver**: tv_fista

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate scatter-to-primary ratio, heel effect profile |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate scatter-to-primary ratio, heel effect profile |
| **M2** Compound | Compound parameter identification (3+ params): Estimate scatter-to-primary ratio, heel effect profile |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate scatter-to-primary ratio, heel effect profile |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate scatter-to-primary ratio, heel effect profile |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct scatter, MTF correction |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct scatter, MTF correction |
| **M2** Compound | Compound correction with rho measurement: Correct scatter, MTF correction |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct scatter, MTF correction |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct scatter, MTF correction |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
