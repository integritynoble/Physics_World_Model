# Coded Aperture Snapshot Spectral Imaging (CASSI) — GAP-TV (200 iter)

**CPU**  **PSNR**: ~24.9 dB  *Yuan et al. 2016 — ~24.9 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'iters': 200, 'lam': 0.01, 'tv_iter': 5}
x = run_solver('best_quality', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
