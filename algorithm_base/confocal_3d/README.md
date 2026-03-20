# Confocal 3D Z-Stack (`confocal_3d`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | 3D Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | 3D CARE | `pwm_core.recon.care_unet.run_care` | Yes |  |
| `famous_dl` | CARE-3D | `pwm_core.recon.care_unet.run_care` | No |  |
| `small_gpu` | CARE-3D (slice-wise) | `pwm_core.recon.care_unet.run_care` | No |  |

## Usage

```python
# Import and run
from algorithm_base.confocal_3d import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.confocal_3d import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CARE 3D | 2018 | 32.0 | 59.8 | done |
| Noise2Void 3D | 2019 | 28.0 | 59.8 | done |
| CARE-3D (PWM) | — | 27.3 | 59.8 | done |
| CARE-3D (slice-wise) (PWM) | — | 27.3 | 59.8 | done |
| precomputed_baseline (test) | — | 27.3 | 59.8 | done |
| rl_20iter (test) | — | 27.3 | 59.8 | done |
| Richardson-Lucy 3D | 1972 | 26.0 | 59.8 | done |
