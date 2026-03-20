# Radio Interferometry (VLBI) — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: UV-plane data (N_baselines, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/radio_interferometry/public/`

```python
from algorithm_base.radio_interferometry.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
