# Neutron Radiography / Tomography (`neutron_tomo`)

**Category**: Scientific Instrumentation | **Canonical DAG**: Pi --> D | **Carrier**: Neutron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: filtered_back_projection

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Beam flux, collimation ratio, rotation steps |
| **M1** Synthetic | Prompt tested with synthetic data validation: Beam flux, collimation ratio, rotation steps |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Beam flux, collimation ratio, rotation steps |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Beam flux, collimation ratio, rotation steps |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Neutron Radiography / Tomography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: FBP under beam hardening, scattering, gamma |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: FBP under beam hardening, scattering, gamma |
| **M2** Compound | Compound mismatch (3+ params simultaneously): FBP under beam hardening, scattering, gamma |
| **M3** Real Data | Real experimental data with measured mismatch: FBP under beam hardening, scattering, gamma |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: FBP under beam hardening, scattering, gamma |

### Mismatch Parameters
Pi→D. Beam spectrum +/-10%, scattering [0,15%], gamma [0,5%].

### Solvers & Expected Performance
- **Solver**: filtered_back_projection

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate beam spectrum, scattering factor |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate beam spectrum, scattering factor |
| **M2** Compound | Compound parameter identification (3+ params): Estimate beam spectrum, scattering factor |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate beam spectrum, scattering factor |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate beam spectrum, scattering factor |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct beam hardening, scatter, gamma |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct beam hardening, scatter, gamma |
| **M2** Compound | Compound correction with rho measurement: Correct beam hardening, scatter, gamma |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct beam hardening, scatter, gamma |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct beam hardening, scatter, gamma |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
