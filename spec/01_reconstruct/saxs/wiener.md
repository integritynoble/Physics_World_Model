# Small-Angle X-ray Scattering (SAXS) — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: scattering pattern (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/saxs/public/`

```python
from algorithm_base.saxs.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
