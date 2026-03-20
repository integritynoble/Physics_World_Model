# Single-Pixel Camera (SPC) — Tikhonov Regularization

**CPU**  *Tikhonov 1963; Hansen, SIAM 1998*
**Input**: photon counts (T × H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/`

```python
from algorithm_base.spc.solvers import run_solver
x = run_solver('tikhonov', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
