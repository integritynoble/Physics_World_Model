# Full-Waveform Inversion (FWI) (`fwi`)

Category: Broader Experimental Science

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `fwi_dl` | InversionNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.fwi import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.fwi import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| FCNVMB | 2021 | 32.0 | 32.4 | done |
| OpenFWI benchmark | 2022 | 30.0 | 32.4 | done |
| Conventional FWI (gradient descent) | 2009 | 28.4 | 32.4 | done |
| InversionNet | 2020 | 28.0 | 32.4 | done |
| VelocityGAN | 2020 | 26.5 | 32.4 | done |
| Adjoint-state FWI | 2006 | 25.0 | 32.4 | done |
| PnP-ADMM [proxy] (PWM) | — | 15.2 | 32.4 | done |
| precomputed_baseline (test) | — | 12.4 | 32.4 | done |
