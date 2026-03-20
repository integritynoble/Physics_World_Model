# MR Angiography (MRA) — MRA-VesselNet [proxy]

**CPU**
**Input**: k-space (kx × ky × kz, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mra/public/`

```python
from algorithm_base.mra.solvers import run_solver
x = run_solver('mra_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
