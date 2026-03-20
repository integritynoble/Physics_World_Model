# Electron Tomography — SIRT-3D [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: tilt series (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_tomography/public/`

```python
from algorithm_base.electron_tomography.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
