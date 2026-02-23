# Seismic Tomography (`seismic_tomo`)

**Category**: Broader Experimental Science | **Canonical DAG**: P --> D | **Carrier**: Seismic
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: travel_time_inversion

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Station array, frequency band, ray geometry |
| **M1** Synthetic | Prompt tested with synthetic data validation: Station array, frequency band, ray geometry |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Station array, frequency band, ray geometry |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Station array, frequency band, ray geometry |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Seismic Tomography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Travel-time / FWI under velocity model error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Travel-time / FWI under velocity model error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Travel-time / FWI under velocity model error |
| **M3** Real Data | Real experimental data with measured mismatch: Travel-time / FWI under velocity model error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Travel-time / FWI under velocity model error |

### Mismatch Parameters
P→D. Velocity +/-10%, source location [0,5] km.

### Solvers & Expected Performance
- **Solver**: travel_time_inversion

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate velocity structure, source locations |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate velocity structure, source locations |
| **M2** Compound | Compound parameter identification (3+ params): Estimate velocity structure, source locations |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate velocity structure, source locations |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate velocity structure, source locations |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct velocity model, relocate sources |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct velocity model, relocate sources |
| **M2** Compound | Compound correction with rho measurement: Correct velocity model, relocate sources |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct velocity model, relocate sources |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct velocity model, relocate sources |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
