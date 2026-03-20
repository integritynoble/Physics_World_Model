# CEST MRI — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: Z-spectrum (offsets × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cest_mri/public/`

```python
from algorithm_base.cest_mri.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
