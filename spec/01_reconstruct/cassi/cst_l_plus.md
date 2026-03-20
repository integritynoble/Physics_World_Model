# Coded Aperture Snapshot Spectral Imaging (CASSI) — CST-L-Plus

**GPU**  **PSNR**: ~36.1 dB  *Cai et al., ECCV 2022 — 36.1 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'model_key': 'cst_l_plus'}
x = run_solver('cst_l_plus', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
