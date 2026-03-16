# Confocal Live-Cell Microscopy (`confocal_livecell`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes |  |
| `famous_dl` | CARE | `pwm_core.recon.care_unet.run_care` | No |  |
| `small_gpu` | CARE | `pwm_core.recon.care_unet.run_care` | No |  |

## Usage

```python
# Import and run
from algorithm_base.confocal_livecell import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.confocal_livecell import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CARE | 2018 | 33.0 | 60.9 | done |
| precomputed_baseline (test) | — | 32.3 | 60.9 | done |
| Noise2Void | 2019 | 29.0 | 60.9 | done |
| Richardson-Lucy | 1972 | 28.0 | 60.9 | done |
