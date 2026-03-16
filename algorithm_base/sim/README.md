# Structured Illumination Microscopy (SIM) (`sim`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Wiener-SIM | `pwm_core.recon.sim_solver.run_sim_reconstruction` | No |  |
| `best_quality` | HiFi-SIM | `pwm_core.recon.sim_solver.run_sim_reconstruction` | No | Wen et al. 2021, Light: S&A |
| `famous_dl` | fairSIM (open-source) | `pwm_core.recon.sim_solver.run_sim_reconstruction` | No | Mueller et al. 2016, Nature Comm. |
| `small_gpu` | Wiener-SIM (fast) | `pwm_core.recon.sim_solver.run_sim_reconstruction` | No |  |

## Usage

```python
# Import and run
from algorithm_base.sim import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.sim import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| ML-SIM | 2021 | 33.0 | 19.5 | gap |
| fairSIM | 2015 | 30.5 | 19.5 | gap |
| Wiener-SIM | 2008 | 30.0 | 19.5 | gap |
| HiFi-SIM (PWM) | — | 24.0 | 19.5 | partial |
| Wiener-SIM (fast) (PWM) | — | 24.0 | 19.5 | partial |
| precomputed_baseline (test) | — | 24.0 | 19.5 | partial |
| wiener_sim (test) | — | 24.0 | 19.5 | partial |
| Bicubic interpolation | 2000 | 22.0 | 19.5 | done |
