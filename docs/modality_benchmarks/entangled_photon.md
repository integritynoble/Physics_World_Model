# Entangled Photon Microscopy (`entangled_photon`)

**Category**: Quantum Imaging | **Canonical DAG**: M --> R --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: coincidence_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Photon pair source, coincidence window, NA |
| **M1** Synthetic | Prompt tested with synthetic data validation: Photon pair source, coincidence window, NA |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Photon pair source, coincidence window, NA |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Photon pair source, coincidence window, NA |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Entangled Photon Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Coincidence image under accidentals, dark counts |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Coincidence image under accidentals, dark counts |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Coincidence image under accidentals, dark counts |
| **M3** Real Data | Real experimental data with measured mismatch: Coincidence image under accidentals, dark counts |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Coincidence image under accidentals, dark counts |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Pair generation rate | optimal | [0.1x, 10x] | - |
| Coincidence window | 1 | [0.1, 10] | ns |
| Accidental coincidence rate | 0 | [0, 20%] of real | - |
| Photon loss (per arm) | 0 | [0, 6] | dB |

### Solvers & Expected Performance
- **Solver**: coincidence_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate pair rate, accidental fraction, timing jitter |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate pair rate, accidental fraction, timing jitter |
| **M2** Compound | Compound parameter identification (3+ params): Estimate pair rate, accidental fraction, timing jitter |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate pair rate, accidental fraction, timing jitter |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate pair rate, accidental fraction, timing jitter |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct accidentals, timing calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct accidentals, timing calibration |
| **M2** Compound | Compound correction with rho measurement: Correct accidentals, timing calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct accidentals, timing calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct accidentals, timing calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
