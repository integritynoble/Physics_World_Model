# Track 4: Cross-Modal — Transfer Across Modality Families

## Challenge

Demonstrate that a single solver or calibration strategy generalizes across
multiple modality families without modality-specific tuning.

## Modalities

Participants are evaluated on **all available modalities** simultaneously:
- Spectral: CASSI, SPC
- Temporal: CACTI
- Microscopy: widefield, confocal, lightsheet
- Medical: MRI, CT (when available)
- Other: as added to the PWM registry

## Scoring

Cross-modal score is the **harmonic mean** of per-modality rho values:

```
rho_cross = N / sum(1/rho_i for i in modalities)
```

The harmonic mean penalizes solvers that fail on any single modality,
rewarding true generalization over specialization.

## Rules

1. **Same solver configuration** across all modalities (no per-modality tuning)
2. Solver config.yaml must declare `supported_modalities: ["*"]`
3. Calibrator may adapt per-modality (calibration IS the adaptive step)
4. Compute budget is per-modality (not shared)

## Evaluation

- Primary: rho_cross (harmonic mean of per-modality rho)
- Secondary: number of modalities where rho > 0.50
- Bonus: number of modalities where rho > 0.80

## Submission

```bash
# Run on all modalities
for mod in cassi spc cacti widefield; do
    pwm evaluate --modality $mod --solver my_solver --output submission/$mod/
done
pwm submit submission/
```
