# Expansion Microscopy (ExM) (`expansion`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `exm_dl` | EXpansionNet | `pwm_core.recon.expansion_solvers.expansion_dl_recon` | Yes | Weigert, M. et al. (2018) CARE for fluorescence microscopy, Nature Methods 15:1090 |

## Usage

```python
# Import and run
from algorithm_base.expansion import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.expansion import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CARE (PWM) | — | 33.9 | 34.0 | done |
| EXpansionNet (PWM) | — | 33.9 | 34.0 | done |
| precomputed_baseline (test) | — | 33.9 | 34.0 | done |
| Noise2Void | 2019 | 28.0 | 34.0 | done |
| Richardson-Lucy ExM | 2015 | 26.0 | 34.0 | done |
