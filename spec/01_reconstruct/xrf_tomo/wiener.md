# X-ray Fluorescence Tomography — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: XRF sinograms (elem × angles × det, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xrf_tomo/public/`

```python
from algorithm_base.xrf_tomo.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
