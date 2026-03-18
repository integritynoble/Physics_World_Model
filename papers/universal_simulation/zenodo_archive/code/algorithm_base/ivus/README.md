# Intravascular Ultrasound (IVUS) (`ivus`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `ivus_dl` | IVUS-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.ivus import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ivus import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| IVUS-Net | 2020 | 30.0 | 26.3 | partial |
| U-Net segmentation | 2020 | 25.0 | 26.3 | done |
| DAS beamforming | 1990 | 22.0 | 26.3 | done |
| FBP [proxy] (PWM) | — | 20.8 | 26.3 | done |
| DL-Recon [proxy] (PWM) | — | 20.8 | 26.3 | done |
| precomputed_baseline (test) | — | 19.8 | 26.3 | done |
