# Transmission Electron Microscopy (TEM) — 3D-CNN

**GPU**  *3D CNN reconstruction, 2018*
**Input**: TEM image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tem/public/`

```python
from algorithm_base.tem.solvers import run_solver
x = run_solver('dl_3dcnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
