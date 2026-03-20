# PET/MR Fusion — Adjoint [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: PET sino + MRI k-space (both float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pet_mr/public/`

```python
from algorithm_base.pet_mr.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
