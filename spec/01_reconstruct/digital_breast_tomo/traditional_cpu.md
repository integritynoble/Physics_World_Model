# Digital Breast Tomosynthesis (DBT) — FBP [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/digital_breast_tomo/public/`

```python
from algorithm_base.digital_breast_tomo.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
