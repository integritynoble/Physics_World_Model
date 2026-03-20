# Ocean Acoustic Tomography — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: travel times (pairs, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ocean_acoustic_tomo/public/`

```python
from algorithm_base.ocean_acoustic_tomo.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
