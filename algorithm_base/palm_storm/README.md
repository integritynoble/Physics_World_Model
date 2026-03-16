# PALM/STORM Single-Molecule Localization (`palm_storm`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy (STORM/PALM) | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | DECODE-SMLM | `pwm_core.recon.smlm_solvers.decode_smlm_recon` | Yes | Speiser, A. et al. (2021) Deep learning enables fast and dense SMLM, Nature Methods 18:1090 |
| `famous_dl` | DeepSTORM | `pwm_core.recon.smlm_solvers.deep_storm_recon` | Yes | Nehme, E. et al. (2018) Deep-STORM: super-resolution microscopy, Optica 5(4) |

## Usage

```python
# Import and run
from algorithm_base.palm_storm import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.palm_storm import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy (STORM/PALM) (PWM) | — | 32.4 | 37.9 | done |
| DECODE-SMLM (PWM) | — | 32.4 | 37.9 | done |
| DeepSTORM (PWM) | — | 32.4 | 37.9 | done |
| precomputed_baseline (test) | — | 32.4 | 37.9 | done |
| rl_20iter (test) | — | 32.4 | 37.9 | done |
| DECODE | 2021 | 25.0 | 37.9 | done |
| Deep-STORM | 2018 | 22.0 | 37.9 | done |
| ThunderSTORM | 2014 | 18.0 | 37.9 | done |
