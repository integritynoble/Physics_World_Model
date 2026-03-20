# Cryo-Electron Tomography (Cryo-ET) — Richardson-Lucy

**CPU**
**Input**: tilt series (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_et/public/`

```python
from algorithm_base.cryo_et.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
