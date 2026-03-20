# Track 1: Correct — Spectral Imaging Under Mismatch

## Challenge

Reconstruct hyperspectral images from CASSI measurements when the assumed
forward model differs from reality.

## Modality

**CASSI** (Coded Aperture Snapshot Spectral Imager)
- 28 spectral channels
- Coded aperture mask with spectral dispersion
- Mismatch sources: mask shift, dispersion calibration, spectral response drift

## Scenarios

Participants must provide a solver that handles all 4 scenarios:

| Scenario | Operator | What It Tests |
|----------|----------|--------------|
| I | H_true (ground truth) | Upper bound on reconstruction quality |
| II | H_nom (mismatched) | Baseline: what happens without calibration |
| III | H_hat (calibrated) | **The test**: calibrate then reconstruct |
| IV | Oracle mask | Diagnostic: how much mismatch matters |

## Mismatch Conditions

| Severity | Mask Shift (px) | Dispersion Error (%) | Spectral Drift (nm) |
|----------|----------------|---------------------|---------------------|
| Mild | 0.5 | 2% | 1.0 |
| Moderate | 1.5 | 5% | 3.0 |
| Severe | 3.0 | 10% | 5.0 |

Evaluation uses **moderate** severity for ranking.

## Evaluation

- Primary: rho (recovery ratio)
- Secondary: oracle_gap (dB), RoIC (dB/GPU-hour)
- Safety brakes: rho < 0.30 = blocked, budget > 2x declared = DQ

## Submission

```bash
pwm evaluate --modality cassi --solver my_solver --track correct \
    --severity moderate --scenes 10 --output submission/
pwm submit submission/
```
