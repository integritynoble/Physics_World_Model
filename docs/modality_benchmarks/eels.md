# Electron Energy Loss Spectroscopy (EELS) (`eels`)

**Category**: Electron Microscopy | **Canonical DAG**: S --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: fourier_ratio

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Energy range, dispersion, collection angle |
| **M1** Synthetic | Prompt tested with synthetic data validation: Energy range, dispersion, collection angle |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Energy range, dispersion, collection angle |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Energy range, dispersion, collection angle |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Electron Energy Loss Spectroscopy (EELS) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Fourier ratio under energy drift, gain variation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Fourier ratio under energy drift, gain variation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Fourier ratio under energy drift, gain variation |
| **M3** Real Data | Real experimental data with measured mismatch: Fourier ratio under energy drift, gain variation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Fourier ratio under energy drift, gain variation |

### Mismatch Parameters
S→D, Electron. Energy drift [-2,2] eV, gain instability [0,5%].

### Solvers & Expected Performance
- **Solver**: fourier_ratio

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for S --> D: Estimate energy drift, gain instability |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate energy drift, gain instability |
| **M2** Compound | Compound parameter identification (3+ params): Estimate energy drift, gain instability |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate energy drift, gain instability |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate energy drift, gain instability |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct energy calibration, gain |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct energy calibration, gain |
| **M2** Compound | Compound correction with rho measurement: Correct energy calibration, gain |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct energy calibration, gain |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct energy calibration, gain |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
