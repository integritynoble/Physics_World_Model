# Photometric Stereo — Unrolled-Net

**GPU**  *Deep unrolling for CS, 2020*
**Input**: images under N lights (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/photometric_stereo/public/`

```python
from algorithm_base.photometric_stereo.solvers import run_solver
x = run_solver('dl_unrolled', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
