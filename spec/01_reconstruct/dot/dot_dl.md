# Diffuse Optical Tomography (DOT) — DOT-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: boundary flux (sources × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dot/public/`

```python
from algorithm_base.dot.solvers import run_solver
x = run_solver('dot_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
