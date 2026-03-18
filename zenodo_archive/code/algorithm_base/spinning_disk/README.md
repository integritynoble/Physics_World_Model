# Spinning Disk Confocal Microscopy (`spinning_disk`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `sd_dl` | SD-CARE | `pwm_core.recon.spinning_disk_solvers.sd_care_recon` | Yes | Weigert, M. et al. (2018) CARE for spinning disk confocal, Nature Methods 15:1090 |

## Usage

```python
# Import and run
from algorithm_base.spinning_disk import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.spinning_disk import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CARE | 2018 | 32.0 | 43.2 | done |
| SD-CARE (PWM) | — | 30.6 | 43.2 | done |
| precomputed_baseline (test) | — | 30.6 | 43.2 | done |
| Richardson-Lucy | 1972 | 27.0 | 43.2 | done |
