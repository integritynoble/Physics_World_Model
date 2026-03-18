# Neutron Radiography / Tomography (`neutron_tomo`)

Category: Scientific Instrumentation

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP (neutron tomography) | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | NeuTomo-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | GRIDREC-Neutron [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.neutron_tomo import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.neutron_tomo import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SIRT | 1972 | 28.0 | 33.4 | done |
| FBP | 1971 | 25.0 | 33.4 | done |
| NeuTomo-DL [proxy] (PWM) | — | 8.7 | 33.4 | done |
| GRIDREC-Neutron [proxy] (PWM) | — | 8.7 | 33.4 | done |
| precomputed_baseline (test) | — | 6.6 | 33.4 | done |
