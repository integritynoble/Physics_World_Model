# Ptychographic Imaging (`ptychography`)

Category: Coherent Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | ePIE | `pwm_core.recon.ptychography_solver.run_epie` | No |  |
| `best_quality` | PtychoNN | `pwm_core.recon.ptychonn.run_ptychonn` | No | Cherukara et al. 2020 |
| `famous_dl` | PtychoNN | `pwm_core.recon.ptychonn.run_ptychonn` | No |  |
| `small_gpu` | PtychoNN 2.0 | `pwm_core.recon.ptychonn.run_ptychonn` | No |  |

## Usage

```python
# Import and run
from algorithm_base.ptychography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ptychography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| AutoPhaseNN | 2022 | 33.0 | 17.6 | gap |
| PtychoNN | 2020 | 31.0 | 17.6 | gap |
| ePIE | 2009 | 28.0 | 17.6 | gap |
| PIE | 2004 | 22.0 | 17.6 | partial |
| PtychoNN 2.0 (PWM) | — | 21.0 | 17.6 | partial |
| precomputed_baseline (test) | — | 21.0 | 17.6 | partial |
| precomputed_phase_baseline (test) | — | 21.0 | 17.6 | partial |
