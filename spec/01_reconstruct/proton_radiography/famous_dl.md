# Proton Radiography — FBP-Proton [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: fluence map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/proton_radiography/public/`

```python
from algorithm_base.proton_radiography.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
