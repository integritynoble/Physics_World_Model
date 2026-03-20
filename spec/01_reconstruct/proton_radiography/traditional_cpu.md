# Proton Radiography — FBP (proton radiography)

**CPU**
**Input**: fluence map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/proton_radiography/public/`

```python
from algorithm_base.proton_radiography.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
