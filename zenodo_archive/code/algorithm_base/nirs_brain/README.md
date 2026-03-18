# Functional Near-Infrared Spectroscopy (fNIRS) (`nirs_brain`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `nirs_dl` | fNIRS-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.nirs_brain import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.nirs_brain import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CNN-LSTM Hybrid | 2024 | 32.1 | 19.3 | gap |
| OT-NIRS (tomographic) | 2010 | 22.0 | 19.3 | done |
| FBP [proxy] (PWM) | — | 21.4 | 19.3 | done |
| DL-Recon [proxy] (PWM) | — | 21.4 | 19.3 | done |
| fNIRS-Net [proxy] (PWM) | — | 21.4 | 19.3 | done |
| precomputed_baseline (test) | — | 20.2 | 19.3 | done |
| MBLL | 1988 | 20.0 | 19.3 | done |
