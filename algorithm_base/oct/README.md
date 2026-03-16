# Optical Coherence Tomography (OCT) (`oct`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FFT Recon | `pwm_core.recon.oct_solver.run_oct` | No |  |
| `best_quality` | Spectral Estimation | `pwm_core.recon.oct_solver.spectral_estimation_recon` | No | Leitgeb et al. 2003, Optics Express |
| `famous_dl` | OCT Denoising Net | `pwm_core.recon.oct_solver.oct_denoising_net_recon` | No | Devalla et al. 2019, Biomed. Optics Express |
| `small_gpu` | OCT Denoising Net | `pwm_core.recon.oct_solver.oct_denoising_net_recon` | No |  |

## Usage

```python
# Import and run
from algorithm_base.oct import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.oct import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SwinIR | 2021 | 35.0 | 21.0 | gap |
| PSCAT | 2022 | 32.2 | 21.0 | gap |
| BM3D | 2007 | 25.0 | 21.0 | partial |
| FFT Recon (PWM) | — | 23.5 | 21.0 | done |
| Spectral Estimation (PWM) | — | 23.5 | 21.0 | done |
| OCT Denoising Net (PWM) | — | 23.5 | 21.0 | done |
| bscan_baseline (test) | — | 23.5 | 21.0 | done |
| bscan_ideal_baseline (test) | — | 23.5 | 21.0 | done |
