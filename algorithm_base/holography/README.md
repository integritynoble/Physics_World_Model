# Digital Holographic Microscopy (`holography`)

Category: Coherent Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Angular Spectrum | `pwm_core.recon.holography_solver.run_holography_reconstruction` | No |  |
| `best_quality` | PhaseNet | `pwm_core.recon.phasenet.run_phasenet` | No | Rivenson et al. 2018, Light: S&A |
| `famous_dl` | PhaseNet | `pwm_core.recon.phasenet.run_phasenet` | No |  |
| `small_gpu` | PhaseNet | `pwm_core.recon.phasenet.run_phasenet` | No |  |

## Usage

```python
# Import and run
from algorithm_base.holography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.holography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Phase distortion DL | 2024 | 36.9 | 22.7 | gap |
| CEHAN (CGH) | 2025 | 35.7 | 22.7 | gap |
| Wirtinger Holography | 2020 | 30.0 | 22.7 | partial |
| HIO | 1982 | 25.0 | 22.7 | done |
| Angular Spectrum | 2000 | 22.0 | 22.7 | done |
| GS (Gerchberg-Saxton) | 1972 | 20.0 | 22.7 | done |
| Direct backpropagation | 1970 | 15.0 | 22.7 | done |
| sqrt_intensity_amplitude (test) | — | 14.9 | 22.7 | done |
