# Magnetic Particle Imaging (MPI) (`magnetic_particle`)

Category: Broader Experimental Science

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `mpi_dl` | MPI-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.magnetic_particle import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.magnetic_particle import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| VRF-Net (recon) | 2026 | 41.6 | 32.5 | partial |
| SRCNN (MPI) | 2024 | 32.9 | 32.5 | done |
| Hybrid encoder-decoder | 2025 | 29.1 | 32.5 | done |
| Adjoint [proxy] (PWM) | — | 27.5 | 32.5 | done |
| PnP-ADMM [proxy] (PWM) | — | 27.5 | 32.5 | done |
| MPI-Net [proxy] (PWM) | — | 27.5 | 32.5 | done |
| precomputed_baseline (test) | — | 26.5 | 32.5 | done |
| X-space approach | 2010 | 26.0 | 32.5 | done |
| System matrix reconstruction | 2005 | 22.0 | 32.5 | done |
