# Coded Aperture Snapshot Spectral Imaging (CASSI) — GAP-TV

**CPU**  **PSNR**: ~24.34 dB  *Yuan et al. 2016 — 24.34 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'iters': 100, 'lam': 0.1, 'tv_iter': 5}
x = run_solver('traditional_cpu', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
