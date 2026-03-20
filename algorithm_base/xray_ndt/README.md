# X-ray NDT (Radiography) (`xray_ndt`)

Category: Industrial Inspection

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `ndt_dl` | NDT-DefectNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.xray_ndt import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.xray_ndt import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| U-Net++ | 2025 | 32.3 | 43.0 | done |
| BM3D denoising | 2007 | 32.0 | 43.0 | done |
| FBP | 1971 | 28.0 | 43.0 | done |
| Raw projection (no filtering) | 2000 | 18.0 | 43.0 | done |
| Adjoint [proxy] (PWM) | — | 17.7 | 43.0 | done |
| PnP-ADMM [proxy] (PWM) | — | 17.7 | 43.0 | done |
| NDT-DefectNet [proxy] (PWM) | — | 17.7 | 43.0 | done |
| precomputed_baseline (test) | — | 16.7 | 43.0 | done |
