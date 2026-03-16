# MR Elastography (MRE) (`mr_elastography`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `mre_dl` | MRE-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.mr_elastography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.mr_elastography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SW-ViT (simulated) | 2025 | 32.7 | 25.1 | partial |
| Phase gradient | 2001 | 24.0 | 25.1 | done |
| Direct inversion | 2001 | 22.0 | 25.1 | done |
| FBP [proxy] (PWM) | — | 13.0 | 25.1 | done |
| DL-Recon [proxy] (PWM) | — | 13.0 | 25.1 | done |
| MRE-Net [proxy] (PWM) | — | 13.0 | 25.1 | done |
| precomputed_baseline (test) | — | 11.0 | 25.1 | done |
