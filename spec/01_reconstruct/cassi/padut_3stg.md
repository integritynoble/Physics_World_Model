# Coded Aperture Snapshot Spectral Imaging (CASSI) — PADUT-3stg

**GPU**  **PSNR**: ~36.95 dB  *Li et al., ICCV 2023 — 36.95 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'model_key': 'padut_3stg'}
x = run_solver('padut_3stg', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
