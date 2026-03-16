# OCT Angiography (OCTA) (`octa`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FFT Recon (OCTA) | `pwm_core.recon.oct_solver.run_oct` | No |  |
| `best_quality` | OCTA-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | OCTA-FF [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.octa import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.octa import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Motion artifact DL | 2024 | 32.7 | 20.9 | gap |
| SU-Net (Siamese) | 2019 | 28.0 | 20.9 | partial |
| CNN accelerated OCTA | 2022 | 20.8 | 20.9 | done |
| OCTA-Net [proxy] (PWM) | — | 20.2 | 20.9 | done |
| OCTA-FF [proxy] (PWM) | — | 20.2 | 20.9 | done |
| FFT Recon (OCTA) (PWM) | — | 18.8 | 20.9 | done |
| precomputed_baseline (test) | — | 18.8 | 20.9 | done |
| SSADA (single-scan) | 2012 | 12.1 | 20.9 | done |
| Single-scan OCTA (noisy) | 2021 | 12.1 | 20.9 | done |
