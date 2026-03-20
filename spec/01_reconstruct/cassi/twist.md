# Coded Aperture Snapshot Spectral Imaging (CASSI) — TwIST

**CPU**  **PSNR**: ~23.1 dB  *Bioucas-Dias & Figueiredo, TIP 2007 — 23.1 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'iters': 100, 'lam': 0.01, 'tv_iter': 5}
x = run_solver('twist', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
