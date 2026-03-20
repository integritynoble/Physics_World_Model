# Cryo-EM Single Particle Analysis (`cryo_em`)

Category: Scientific Instrumentation

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `relion_dl` | CryoDRGN [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.cryo_em import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.cryo_em import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Topaz-Denoise | 2020 | 25.0 | 24.7 | done |
| DUAL (cryo-ET) | 2024 | 21.3 | 24.7 | done |
| DRA (denoising-recon) | 2024 | 20.2 | 24.7 | done |
| Adjoint [proxy] (PWM) | — | 20.2 | 24.7 | done |
| PnP-ADMM [proxy] (PWM) | — | 20.2 | 24.7 | done |
| CryoDRGN [proxy] (PWM) | — | 20.2 | 24.7 | done |
| cryoSPARC | 2017 | 20.0 | 24.7 | done |
| precomputed_wiener (test) | — | 19.2 | 24.7 | done |
| rl_ctf_20iter (test) | — | 19.2 | 24.7 | done |
| RELION | 2012 | 18.0 | 24.7 | done |
