# Small-Angle X-ray Scattering (SAXS) (`saxs`)

**Category**: Scientific Instrumentation | **Canonical DAG**: R --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: desmearing

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Beam size, q-range, sample-detector distance |
| **M1** Synthetic | Prompt tested with synthetic data validation: Beam size, q-range, sample-detector distance |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Beam size, q-range, sample-detector distance |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Beam size, q-range, sample-detector distance |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Small-Angle X-ray Scattering (SAXS) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Desmearing under beam divergence, parasitic scatter |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Desmearing under beam divergence, parasitic scatter |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Desmearing under beam divergence, parasitic scatter |
| **M3** Real Data | Real experimental data with measured mismatch: Desmearing under beam divergence, parasitic scatter |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Desmearing under beam divergence, parasitic scatter |

### Mismatch Parameters
R→D. Beam divergence [0.05,0.5] mrad, parasitic scatter [0,20%].

### Solvers & Expected Performance
- **Solver**: desmearing

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for R --> D: Estimate beam profile, background scatter |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate beam profile, background scatter |
| **M2** Compound | Compound parameter identification (3+ params): Estimate beam profile, background scatter |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate beam profile, background scatter |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate beam profile, background scatter |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct beam smearing, background |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct beam smearing, background |
| **M2** Compound | Compound correction with rho measurement: Correct beam smearing, background |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct beam smearing, background |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct beam smearing, background |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
