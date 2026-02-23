# Muon Tomography (`muon_tomo`)

**Category**: Scientific Instrumentation | **Canonical DAG**: Pi --> D | **Carrier**: Muon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: poca_reconstruction

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Detector layers, angular resolution, integration time |
| **M1** Synthetic | Prompt tested with synthetic data validation: Detector layers, angular resolution, integration time |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Detector layers, angular resolution, integration time |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Detector layers, angular resolution, integration time |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Muon Tomography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: POCA / MLP under angular resolution limit |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: POCA / MLP under angular resolution limit |
| **M2** Compound | Compound mismatch (3+ params simultaneously): POCA / MLP under angular resolution limit |
| **M3** Real Data | Real experimental data with measured mismatch: POCA / MLP under angular resolution limit |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: POCA / MLP under angular resolution limit |

### Mismatch Parameters
Pi→D. Angular resolution [3,15] mrad, alignment [0,1] mm.

### Solvers & Expected Performance
- **Solver**: poca_reconstruction

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate detector alignment, angular uncertainty |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate detector alignment, angular uncertainty |
| **M2** Compound | Compound parameter identification (3+ params): Estimate detector alignment, angular uncertainty |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate detector alignment, angular uncertainty |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate detector alignment, angular uncertainty |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct alignment, track fitting |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct alignment, track fitting |
| **M2** Compound | Compound correction with rho measurement: Correct alignment, track fitting |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct alignment, track fitting |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct alignment, track fitting |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
