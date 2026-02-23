# OCT Angiography (OCTA) (`octa`)

**Category**: Medical Imaging | **Canonical DAG**: P+P --> Sigma --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: tv_fista

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Scan density, interscan time, decorrelation method |
| **M1** Synthetic | Prompt tested with synthetic data validation: Scan density, interscan time, decorrelation method |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Scan density, interscan time, decorrelation method |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Scan density, interscan time, decorrelation method |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for OCT Angiography (OCTA) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: TV-FISTA under bulk motion, projection artifact |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: TV-FISTA under bulk motion, projection artifact |
| **M2** Compound | Compound mismatch (3+ params simultaneously): TV-FISTA under bulk motion, projection artifact |
| **M3** Real Data | Real experimental data with measured mismatch: TV-FISTA under bulk motion, projection artifact |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: TV-FISTA under bulk motion, projection artifact |

### Mismatch Parameters
P+P→Sigma→D, Photon. Bulk motion [0,50] um, projection artifacts [0,0.3].

### Solvers & Expected Performance
- **Solver**: tv_fista

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P+P --> Sigma --> D: Estimate bulk motion, shadow artifacts |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate bulk motion, shadow artifacts |
| **M2** Compound | Compound parameter identification (3+ params): Estimate bulk motion, shadow artifacts |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate bulk motion, shadow artifacts |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate bulk motion, shadow artifacts |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct bulk motion, projection artifacts |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct bulk motion, projection artifacts |
| **M2** Compound | Compound correction with rho measurement: Correct bulk motion, projection artifacts |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct bulk motion, projection artifacts |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct bulk motion, projection artifacts |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
