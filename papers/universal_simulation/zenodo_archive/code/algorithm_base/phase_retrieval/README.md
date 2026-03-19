# Coherent Diffractive Imaging / Phase Retrieval (`phase_retrieval`)

Category: Coherent Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | HIO | `pwm_core.recon.phase_retrieval_solver.run_phase_retrieval` | No |  |
| `best_quality` | RAAR [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | prDeep [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `small_gpu` | prDeep [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.phase_retrieval import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.phase_retrieval import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DLMMPR (coded diffraction) | 2025 | 45.8 | 27.4 | gap |
| NAS-PRNet (bio cells) | 2022 | 36.7 | 27.4 | partial |
| WF (Wirtinger Flow) | 2015 | 30.0 | 27.4 | done |
| HIO | 1982 | 25.0 | 27.4 | done |
| ER (Error Reduction) | 1972 | 23.0 | 27.4 | done |
| Wiener (low SNR) | 2000 | 18.0 | 27.4 | done |
| HIO (0 dB input SNR) | 2015 | 14.0 | 27.4 | done |
| RAAR [proxy] (PWM) | — | 13.6 | 27.4 | done |
| prDeep [proxy] (PWM) | — | 13.6 | 27.4 | done |
| precomputed_baseline (test) | — | 12.6 | 27.4 | done |
