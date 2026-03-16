# Photometric Stereo (`photometric_stereo`)

Category: Depth Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `ps_dl` | PS-FCN [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.photometric_stereo import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.photometric_stereo import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CNN-PS | 2019 | 32.0 | 28.7 | partial |
| Adjoint [proxy] (PWM) | — | 30.0 | 28.7 | done |
| PnP-ADMM [proxy] (PWM) | — | 30.0 | 28.7 | done |
| PS-FCN [proxy] (PWM) | — | 30.0 | 28.7 | done |
| precomputed_baseline (test) | — | 29.0 | 28.7 | done |
| Woodham (Lambertian) | 1980 | 25.0 | 28.7 | done |
