# Scanning Electron Microscopy (SEM) — 3D-CNN

**GPU**  *3D CNN reconstruction, 2018*
**Input**: SEM image (H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sem/public/`

```python
from algorithm_base.sem.solvers import run_solver
x = run_solver('dl_3dcnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
