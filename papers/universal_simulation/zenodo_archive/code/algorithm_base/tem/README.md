# Transmission Electron Microscopy (TEM) (`tem`)

Category: Electron Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FISTA-L2 (CTF correction) | `pwm_core.recon.classical.run_fista_l2` | No |  |
| `best_quality` | TEM-DL (ePIE-Net) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | TEM-UNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.tem import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.tem import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CGRDN | 2024 | 37.0 | 40.8 | done |
| SwinIR | 2021 | 35.0 | 40.8 | done |
| Topaz-Denoise | 2020 | 32.0 | 40.8 | done |
| BM3D | 2007 | 30.4 | 40.8 | done |
| TEM-DL (ePIE-Net) [proxy] (PWM) | — | 26.3 | 40.8 | done |
| TEM-UNet [proxy] (PWM) | — | 26.3 | 40.8 | done |
| Wiener filter (basic) | 2013 | 26.0 | 40.8 | done |
| FISTA-L2 (CTF correction) (PWM) | — | 25.3 | 40.8 | done |
| precomputed_baseline (test) | — | 25.3 | 40.8 | done |
| NLM | 2005 | 25.0 | 40.8 | done |
