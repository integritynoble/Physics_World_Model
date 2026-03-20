# SPECT/CT Fusion — PnP-ADMM [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: SPECT proj + CT sino (both float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spect_ct/public/`

```python
from algorithm_base.spect_ct.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
