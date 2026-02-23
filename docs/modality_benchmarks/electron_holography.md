# Electron Holography (`electron_holography`)

**Category**: Electron Microscopy | **Canonical DAG**: P --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: fourier_sideband

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Biprism voltage, fringe spacing, FOV |
| **M1** Synthetic | Prompt tested with synthetic data validation: Biprism voltage, fringe spacing, FOV |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Biprism voltage, fringe spacing, FOV |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Biprism voltage, fringe spacing, FOV |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Electron Holography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Fourier sideband under biprism drift |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Fourier sideband under biprism drift |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Fourier sideband under biprism drift |
| **M3** Real Data | Real experimental data with measured mismatch: Fourier sideband under biprism drift |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Fourier sideband under biprism drift |

### Mismatch Parameters
P→D, Electron. Biprism drift +/-2%, fringe rotation [-1,1] deg.

### Solvers & Expected Performance
- **Solver**: fourier_sideband

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate biprism voltage drift, fringe rotation |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate biprism voltage drift, fringe rotation |
| **M2** Compound | Compound parameter identification (3+ params): Estimate biprism voltage drift, fringe rotation |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate biprism voltage drift, fringe rotation |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate biprism voltage drift, fringe rotation |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct fringe analysis parameters |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct fringe analysis parameters |
| **M2** Compound | Compound correction with rho measurement: Correct fringe analysis parameters |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct fringe analysis parameters |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct fringe analysis parameters |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
