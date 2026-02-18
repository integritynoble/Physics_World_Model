# Frozen Baseline Pack

Official reference baselines that anchor the PWM leaderboard.

## What Baselines Are

Baselines are **frozen** solver configurations whose parameters, seeds, and results
never change.  Every new submission to LIP-Arena is compared against these baselines.

They serve three purposes:

1. **Floor** -- if you can't beat GAP-TV, something is wrong with your pipeline
2. **Calibration** -- baselines span solver families (classical, PnP, deep) so the
   leaderboard covers the full spectrum
3. **Reproducibility** -- RunBundles in each baseline directory are reference artifacts
   with SHA-256 hashes

## Baselines

| Baseline | Family | CASSI rho | SPC rho | Status |
|----------|--------|-----------|---------|--------|
| GAP-TV | Classical (total variation) | 0.60 | -- | Reference |
| FISTA-TV | Classical (FISTA + TV) | -- | 0.81 | Reference |
| MST-L | Deep learning (transformer) | 0.47 | -- | Reference |
| PnP-ADMM | Plug-and-Play (ADMM + denoiser) | -- | -- | Planned |

## Directory Structure

```
baselines/
  README.md                    # This file
  gap_tv/
    config.yaml                # Frozen parameters
    scores.json                # Official scores (rho, oracle_gap, RoIC)
    runbundles/                # Reference RunBundles per modality
  fista_tv/
    config.yaml
    scores.json
    runbundles/
  mst_l/
    config.yaml
    scores.json
    runbundles/
  pnp_admm/
    config.yaml
    scores.json
    runbundles/
```

## Rules

1. Baselines are **frozen**: same parameters, same seeds, same results forever
2. Every new submission is compared against baselines in the leaderboard
3. Baselines are re-run on new modalities as they are added (parameters don't change)
4. Baselines span solver families: classical, PnP, and deep learning
5. Adding a new baseline requires a governance vote (per GOVERNANCE.md)
6. Baselines provide the **Official** plugin tier (per Addition 21)

## How to Use Baselines

```bash
# Compare your solver against all baselines on CASSI
pwm evaluate --modality cassi --solver my_solver
# Baseline scores are automatically shown in the summary table

# Run a specific baseline
pwm evaluate --modality cassi --solver gap_tv
```

## How to Propose a New Baseline

1. Open an RFC (governance lane)
2. Provide RunBundles for at least 3 modalities
3. Demonstrate reproducibility (3 independent runs, same scores)
4. Steward board votes (per GOVERNANCE.md Section 2.3)
5. If approved, baseline is frozen and never modified
