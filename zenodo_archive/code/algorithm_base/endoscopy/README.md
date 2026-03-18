# Fiber Bundle Endoscopy (`endoscopy`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FISTA-L2 (endoscopy) | `pwm_core.recon.classical.run_fista_l2` | No |  |
| `best_quality` | EndoMapper-Net | `pwm_core.recon.endoscopy_solvers.endomapper_recon` | Yes | Ozyoruk, K.B. et al. (2021) EndoMapper, Nat. Mach. Intel. 3 |
| `famous_dl` | AF-SfMLearner | `pwm_core.recon.endoscopy_solvers.af_sfm_learner_recon` | Yes | Shao, S. et al. (2022) Self-supervised depth estimation in endoscopy, MICCAI 2022 |

## Usage

```python
# Import and run
from algorithm_base.endoscopy import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.endoscopy import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SwinIR | 2024 | 36.8 | 26.8 | gap |
| U-Net denoising | 2019 | 28.0 | 26.8 | done |
| Richardson-Lucy | 1972 | 24.0 | 26.8 | done |
| Interpolation baseline | 2000 | 22.0 | 26.8 | done |
| Raw CLE (honeycomb artifact) | 2022 | 20.6 | 26.8 | done |
| Gaussian filter (fiber bundle) | 2023 | 19.0 | 26.8 | done |
| Raw fiber bundle (no processing) | 2019 | 14.6 | 26.8 | done |
| FISTA-L2 (endoscopy) (PWM) | — | 11.8 | 26.8 | done |
| EndoMapper-Net (PWM) | — | 11.8 | 26.8 | done |
| AF-SfMLearner (PWM) | — | 11.8 | 26.8 | done |
| rl_20iter (test) | — | 11.8 | 26.8 | done |
| rl_50iter (test) | — | 11.8 | 26.8 | done |
| precomputed_recon (test) | — | 11.8 | 26.8 | done |
