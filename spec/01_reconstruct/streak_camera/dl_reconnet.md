# Streak Camera Imaging — ReconNet

**GPU**  *DL for CS reconstruction, 2016*
**Input**: streak image (time × space, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/streak_camera/public/`

```python
from algorithm_base.streak_camera.solvers import run_solver
x = run_solver('dl_reconnet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
