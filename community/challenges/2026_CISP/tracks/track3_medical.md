# Track 3: Medical — MRI/CT Under Acquisition Artifacts

## Challenge

Reconstruct medical images from undersampled or artifact-corrupted
measurements when the acquisition model is imperfectly known.

## Modalities

**MRI** (Magnetic Resonance Imaging) and **CT** (Computed Tomography)
- MRI: k-space undersampling with coil sensitivity mismatch
- CT: sparse-view/limited-angle with beam hardening artifacts
- Mismatch sources: coil sensitivity drift, sampling trajectory error, beam spectrum

## Scenarios

| Scenario | Operator | What It Tests |
|----------|----------|--------------|
| I | H_true | Perfect acquisition model |
| II | H_nom | Assumed model (with artifacts) |
| III | H_hat | Calibrated model |
| IV | Oracle | Diagnostic bound |

## Mismatch Conditions (MRI)

| Severity | Coil Sensitivity Error (%) | Trajectory Error (%) | B0 Inhomogeneity (Hz) |
|----------|---------------------------|---------------------|----------------------|
| Mild | 2% | 0.5% | 10 |
| Moderate | 5% | 2.0% | 50 |
| Severe | 10% | 5.0% | 100 |

## Evaluation

- Primary: rho (recovery ratio)
- Clinical relevance: SSIM in regions of interest
- Safety brakes apply (critical for medical imaging)

## Submission

```bash
pwm evaluate --modality mri --solver my_solver --track correct \
    --severity moderate --scenes 10 --output submission/
pwm submit submission/
```

## Note on Clinical Validation

CISP results do NOT constitute clinical validation. Medical imaging solvers
should undergo separate clinical evaluation before any diagnostic use.
See `docs/IP_POLICY.md` Section 5 for data rights considerations.
