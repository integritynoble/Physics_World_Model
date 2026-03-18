# Industrial X-ray CT (`industrial_ct`)

Category: Industrial Inspection

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `ict_dl` | IndustrialCT-Net [proxy] | `pwm_core.recon.ct_solvers.run_fbp` | No | Shepp & Logan 1974 |

## Usage

```python
# Import and run
from algorithm_base.industrial_ct import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.industrial_ct import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| ADMM-TransNet | 2025 | 44.6 | 7.2 | gap |
| SIRT | 1972 | 30.0 | 7.2 | gap |
| FDK | 1984 | 28.0 | 7.2 | gap |
| Adjoint [proxy] (PWM) | — | 21.3 | 7.2 | gap |
| IndustrialCT-Net [proxy] (PWM) | — | 21.3 | 7.2 | gap |
| precomputed_baseline (test) | — | 20.3 | 7.2 | gap |
