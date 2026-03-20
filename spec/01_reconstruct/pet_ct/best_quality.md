# PET/CT Fusion — PnP-ADMM [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: PET sino + CT proj (both float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pet_ct/public/`

```python
from algorithm_base.pet_ct.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
