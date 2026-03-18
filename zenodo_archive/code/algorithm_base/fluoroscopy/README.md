# Fluoroscopy (`fluoroscopy`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP (fluoroscopy) | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | FluoroNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | X-ray CNN [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.fluoroscopy import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.fluoroscopy import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| FluoroNet [proxy] (PWM) | — | 54.9 | 28.5 | gap |
| X-ray CNN [proxy] (PWM) | — | 54.9 | 28.5 | gap |
| FBP (fluoroscopy) (PWM) | — | 44.5 | 28.5 | gap |
| precomputed_baseline (test) | — | 44.5 | 28.5 | gap |
| MSR2AU-Net | 2024 | 39.1 | 28.5 | gap |
| RED-CNN | 2017 | 33.0 | 28.5 | partial |
| Motion compensation | 2000 | 28.0 | 28.5 | done |
