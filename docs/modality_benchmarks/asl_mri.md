# Arterial Spin Labeling (ASL) MRI (`asl_mri`)

**Category**: Medical Imaging | **Canonical DAG**: M --> F --> S --> D | **Carrier**: Spin/RF
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: perfusion_quantification

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Labeling scheme, PLD, background suppression |
| **M1** Synthetic | Prompt tested with synthetic data validation: Labeling scheme, PLD, background suppression |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Labeling scheme, PLD, background suppression |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Labeling scheme, PLD, background suppression |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Arterial Spin Labeling (ASL) MRI |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Perfusion quantification under transit time error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Perfusion quantification under transit time error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Perfusion quantification under transit time error |
| **M3** Real Data | Real experimental data with measured mismatch: Perfusion quantification under transit time error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Perfusion quantification under transit time error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Labeling efficiency | 0.85 | [0.6, 0.95] | - |
| Transit delay | 1.5 | [0.5, 3.0] | s |
| T1 blood error | 0 | [-10%, 10%] | - |

### Solvers & Expected Performance
- **Solver**: perfusion_quantification

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> F --> S --> D: Estimate bolus arrival time, labeling efficiency |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate bolus arrival time, labeling efficiency |
| **M2** Compound | Compound parameter identification (3+ params): Estimate bolus arrival time, labeling efficiency |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate bolus arrival time, labeling efficiency |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate bolus arrival time, labeling efficiency |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct transit time, partial volume |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct transit time, partial volume |
| **M2** Compound | Compound correction with rho measurement: Correct transit time, partial volume |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct transit time, partial volume |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct transit time, partial volume |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
