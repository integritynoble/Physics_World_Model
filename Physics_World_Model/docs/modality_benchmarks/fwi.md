# Full-Waveform Inversion (FWI) (`fwi`)

**Category**: Broader Experimental Science | **Canonical DAG**: P --> D | **Carrier**: Seismic/Acoustic
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: adjoint_state

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Source array, receiver array, frequency band |
| **M1** Synthetic | Prompt tested with synthetic data validation: Source array, receiver array, frequency band |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Source array, receiver array, frequency band |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Source array, receiver array, frequency band |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Full-Waveform Inversion (FWI) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Adjoint-state inversion under starting model error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Adjoint-state inversion under starting model error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Adjoint-state inversion under starting model error |
| **M3** Real Data | Real experimental data with measured mismatch: Adjoint-state inversion under starting model error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Adjoint-state inversion under starting model error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Starting velocity model error | 0 | [-15%, 15%] | - |
| Source wavelet error | 0 | [-10%, 10%] amplitude | - |
| Anelastic attenuation (Q) | infinite | [50, 500] | - |
| Source location error | 0 | [0, 100] | m |

### Solvers & Expected Performance
- **Solver**: adjoint_state

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate velocity model, attenuation, anisotropy |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate velocity model, attenuation, anisotropy |
| **M2** Compound | Compound parameter identification (3+ params): Estimate velocity model, attenuation, anisotropy |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate velocity model, attenuation, anisotropy |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate velocity model, attenuation, anisotropy |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct starting model, cycle-skip mitigation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct starting model, cycle-skip mitigation |
| **M2** Compound | Compound correction with rho measurement: Correct starting model, cycle-skip mitigation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct starting model, cycle-skip mitigation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct starting model, cycle-skip mitigation |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Multi-scale FWI, elastic (multi-parameter) inversion.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
