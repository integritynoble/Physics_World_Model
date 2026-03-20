# Scanning Electron Microscopy (SEM) — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: SEM image (H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sem/public/`

```python
from algorithm_base.sem.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
