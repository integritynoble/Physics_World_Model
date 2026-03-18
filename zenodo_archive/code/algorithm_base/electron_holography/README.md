# Electron Holography (`electron_holography`)

Category: Electron Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Phase Retrieval (HIO) | `pwm_core.recon.phase_retrieval_solver.run_phase_retrieval` | No |  |
| `best_quality` | EH-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | Phase-Sideband [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.electron_holography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.electron_holography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| FIN (Fourier Imager Network) | 2022 | 36.1 | 26.2 | partial |
| HoloPhaseNet (cGAN) | 2022 | 35.3 | 26.2 | partial |
| DNN phase unwrapping | 2021 | 30.0 | 26.2 | partial |
| Fourier filtering | 1993 | 25.0 | 26.2 | done |
| EH-Net [proxy] (PWM) | — | 11.9 | 26.2 | done |
| Phase-Sideband [proxy] (PWM) | — | 11.9 | 26.2 | done |
| Phase Retrieval (HIO) (PWM) | — | 9.5 | 26.2 | done |
| precomputed_baseline (test) | — | 9.5 | 26.2 | done |
