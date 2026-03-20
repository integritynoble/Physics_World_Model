# Stellar Coronagraphy — DL-SpeckleNull [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/coronagraphy/public/`

```python
from algorithm_base.coronagraphy.solvers import run_solver
x = run_solver('speckle_null_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
