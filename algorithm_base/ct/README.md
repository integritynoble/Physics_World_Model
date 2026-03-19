# X-ray Computed Tomography (CT) (`ct`)

Category: Medical Imaging

## Dataset

**LoDoPaB-CT** — real clinical chest CT from LIDC/IDRI database.
- Source: Leuschner et al., Scientific Data 2021 (doi:10.1038/s41597-021-00893-z)
- Zenodo: https://zenodo.org/records/3384092
- License: CC BY 4.0
- Image size: 362x362, parallel beam, 1000 angles, 512 detectors
- 10 standard samples from test split

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP | `pwm_core.recon.ct_solvers.run_fbp` | No | — |
| `best_quality` | FBP + NLM | `algorithm_base.ct.solvers.run_fbp_nlm` | No | Buades et al. 2005 |
| `famous_dl` | RED-CNN | `pwm_core.recon.redcnn.run_redcnn` | No | Chen et al. 2017, IEEE TMI |
| `small_gpu` | RED-CNN | `pwm_core.recon.redcnn.run_redcnn` | No | — |

## Usage

```python
# Import and run
from algorithm_base.ct import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ct import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Verified Solver Performance (LoDoPaB-CT, 362x362, 1000 angles)

| Solver Key | Name | PWM PSNR | Status |
|-----------|------|----------|--------|
| `traditional_cpu` | FBP | 41.02 dB | verified |

## Algorithm Leaderboard (LoDoPaB-CT reference)

| Algorithm | Year | Ref PSNR | Status |
|-----------|------|----------|--------|
| CT-FM | 2024 | 44.1 | — |
| LEARN | 2019 | 43.1 | — |
| Score-CT | 2022 | 43.0 | — |
| DuDoTrans | 2022 | 42.1 | — |
| FBPConvNet | 2017 | 38.5 | — |
| iRadonMAP | 2019 | 36.9 | — |
| Learned Primal-Dual | 2018 | 36.2 | — |
| DOLCE | 2023 | 36.0 | — |
| FBP + U-Net | 2021 | 35.8 | — |
| TV regularization | 2006 | 33.4 | — |
| RED-CNN | 2017 | 33.2 | — |
| FBP (baseline) | 1971 | 30.5 | verified (PWM: 41.02 dB) |
