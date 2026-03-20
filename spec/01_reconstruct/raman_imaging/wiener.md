# Raman Imaging / Microscopy — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: Raman spectra (H × W × wn, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/public/`

```python
from algorithm_base.raman_imaging.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
