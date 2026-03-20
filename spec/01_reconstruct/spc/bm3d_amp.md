# Single-Pixel Camera (SPC) — BM3D-AMP

**CPU**  *Metzler, Maleki & Baraniuk, IEEE TIT 2016*
**Input**: photon counts (T × H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/`

```python
from algorithm_base.spc.solvers import run_solver
x = run_solver('bm3d_amp', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
