# Arterial Spin Labeling (ASL) MRI — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: label-control pairs (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/asl_mri/public/`

```python
from algorithm_base.asl_mri.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
