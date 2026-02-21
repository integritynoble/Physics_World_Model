# Track 2: Temporal — Video Compressive Imaging Under Motion Blur

## Challenge

Reconstruct high-speed video frames from CACTI compressed measurements
when temporal mismatch (motion blur, frame timing) degrades the assumed model.

## Modality

**CACTI** (Coded Aperture Compressive Temporal Imager)
- 8-16 temporal frames per snapshot
- Coded exposure patterns
- Mismatch sources: mask timing jitter, motion blur kernel error, frame rate drift

## Scenarios

| Scenario | Operator | What It Tests |
|----------|----------|--------------|
| I | H_true | Perfect temporal model |
| II | H_nom | Assumed timing (with jitter/drift) |
| III | H_hat | Calibrated temporal model |
| IV | Oracle | Diagnostic bound |

## Mismatch Conditions

| Severity | Timing Jitter (%) | Blur Kernel Error (px) | Frame Drift (%) |
|----------|-------------------|----------------------|----------------|
| Mild | 1% | 0.3 | 0.5% |
| Moderate | 3% | 1.0 | 2.0% |
| Severe | 5% | 2.0 | 5.0% |

## Evaluation

- Primary: rho (recovery ratio)
- Additional: temporal consistency metric (inter-frame SSIM)
- Budget: wall-clock time per video sequence

## Submission

```bash
pwm evaluate --modality cacti --solver my_solver --track correct \
    --severity moderate --scenes 10 --output submission/
pwm submit submission/
```
