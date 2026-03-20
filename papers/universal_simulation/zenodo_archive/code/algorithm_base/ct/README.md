# X-ray Computed Tomography (CT) (`ct`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | PnP-HQS + NLM | `pwm_core.recon.pnp.run_pnp` | No |  |
| `famous_dl` | RED-CNN | `pwm_core.recon.redcnn.run_redcnn` | No | Chen et al. 2017, IEEE TMI |
| `small_gpu` | RED-CNN | `pwm_core.recon.redcnn.run_redcnn` | No |  |

## Usage

```python
# Import and run
from algorithm_base.ct import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ct import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| LEARN | 2019 | 43.1 | 15.1 | gap |
| Score-CT | 2022 | 43.0 | 15.1 | gap |
| DuDoTrans | 2022 | 42.1 | 15.1 | gap |
| FBPConvNet | 2017 | 38.5 | 15.1 | gap |
| iRadonMAP | 2019 | 36.9 | 15.1 | gap |
| Learned Primal-Dual | 2018 | 36.2 | 15.1 | gap |
| DOLCE | 2023 | 36.0 | 15.1 | gap |
| TV regularization | 2006 | 33.4 | 15.1 | gap |
| RED-CNN | 2017 | 33.2 | 15.1 | gap |
| FBP (Ram-Lak) | 1971 | 30.2 | 15.1 | gap |
| FBP (10 angles) | 2021 | 17.1 | 15.1 | done |
| FBP (5 angles) | 2021 | 15.5 | 15.1 | done |
| PnP-HQS + NLM (PWM) | — | 13.8 | 15.1 | done |
| fbp_ramlak (test) | — | 13.8 | 15.1 | done |
| fbp_shepp_logan (test) | — | 13.8 | 15.1 | done |
| sart_10iter (test) | — | 13.8 | 15.1 | done |
| FBP (2 angles, scattering) | 2021 | 13.1 | 15.1 | done |
