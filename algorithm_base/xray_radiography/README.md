# X-ray Radiography (`xray_radiography`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP (X-ray radiography) | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | CheXNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | X-ray UNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.xray_radiography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.xray_radiography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CheXNet [proxy] (PWM) | — | 46.9 | 47.5 | done |
| X-ray UNet [proxy] (PWM) | — | 46.9 | 47.5 | done |
| Improved Restormer | 2025 | 37.3 | 47.5 | done |
| BM3D | 2007 | 32.0 | 47.5 | done |
| Flat-field + simple filter | 2018 | 30.0 | 47.5 | done |
| NLM | 2005 | 28.0 | 47.5 | done |
| FBP (X-ray radiography) (PWM) | — | 27.1 | 47.5 | done |
| precomputed_baseline (test) | — | 27.1 | 47.5 | done |
| Median filter | 2000 | 25.0 | 47.5 | done |
| Noisy input (flat-field only) | 2018 | 24.1 | 47.5 | done |
