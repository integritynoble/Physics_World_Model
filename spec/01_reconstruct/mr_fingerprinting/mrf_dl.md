# MR Fingerprinting (MRF) — MRF-Net [proxy]

**CPU**
**Input**: signal evolution (T × H × W, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mr_fingerprinting/public/`

```python
from algorithm_base.mr_fingerprinting.solvers import run_solver
x = run_solver('mrf_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
