# Scanning Tunneling Microscopy (STM) (`stm`)

Category: Scanning Probe Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `stm_dl` | STM-Net | `pwm_core.recon.stm_solvers.stm_dl_recon` | Yes | Ziatdinov, M. et al. (2021) DL for atomic-level STM, Nat. Mach. Intell. 3:269 |

## Usage

```python
# Import and run
from algorithm_base.stm import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.stm import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DeepSPM | 2020 | 30.0 | 42.6 | done |
| Richardson-Lucy (PWM) | — | 23.3 | 42.6 | done |
| CARE (PWM) | — | 23.3 | 42.6 | done |
| STM-Net (PWM) | — | 23.3 | 42.6 | done |
| precomputed_baseline (test) | — | 23.3 | 42.6 | done |
| Drift correction | 2000 | 22.0 | 42.6 | done |
