# Muon Tomography (`muon_tomo`)

Category: Scientific Instrumentation

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP (muon tomography) | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | POCA-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | EM-POCA [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.muon_tomo import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.muon_tomo import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| EM-POCA [proxy] (PWM) | — | 19.2 | 35.2 | done |
| mu-Net (ConvNeXt U-Net) | 2023 | 17.1 | 35.2 | done |
| PoCA | 2003 | 13.7 | 35.2 | done |
| PoCA (1024 muons) | 2023 | 13.7 | 35.2 | done |
| FBP (muon tomography) (PWM) | — | 13.5 | 35.2 | done |
| precomputed_baseline (test) | — | 13.5 | 35.2 | done |
| Simple FBP (low stats) | 2003 | 8.0 | 35.2 | done |
