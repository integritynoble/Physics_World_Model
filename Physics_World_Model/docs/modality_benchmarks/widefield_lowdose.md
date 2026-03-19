# Low-Dose Widefield Microscopy (`widefield_lowdose`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M1 | **Forward Model**: linear_operator | **Default Solver**: pnp_hqs

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Photon budget vs SNR tradeoff |
| **M1** Synthetic | Prompt tested with synthetic data validation: Photon budget vs SNR tradeoff |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Photon budget vs SNR tradeoff |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Photon budget vs SNR tradeoff |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Low-Dose Widefield Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Denoise under extreme Poisson noise |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Denoise under extreme Poisson noise |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Denoise under extreme Poisson noise |
| **M3** Real Data | Real experimental data with measured mismatch: Denoise under extreme Poisson noise |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Denoise under extreme Poisson noise |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Photon rate alpha | 100 | [10, 500] | photons/px |
| Read noise sigma | 5.0 | [1.0, 15.0] | e- |
| Background | 50 | [10, 200] | counts |
| Dark current | 0.1 | [0.01, 1.0] | e-/px/s |

### Solvers & Expected Performance
- **Solver(s)**: VST+BM3D, CARE, Noise2Void
- **Scenario I PSNR**: 18-25 dB
- **Scenario II drop**: 3-10 dB

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate photon rate, read noise, background |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate photon rate, read noise, background |
| **M2** Compound | Compound parameter identification (3+ params): Estimate photon rate, read noise, background |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate photon rate, read noise, background |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate photon rate, read noise, background |

### True-Spec Parameters
Alpha (87), read noise (4.2), background (63), dark current (0.15)

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct noise model, denoise-then-deconvolve |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct noise model, denoise-then-deconvolve |
| **M2** Compound | Compound correction with rho measurement: Correct noise model, denoise-then-deconvolve |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct noise model, denoise-then-deconvolve |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct noise model, denoise-then-deconvolve |

### Correction Targets
- **Expected rho**: >= 0.70

### Improvement Roadmap
Add spatially-varying background, camera column noise.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
