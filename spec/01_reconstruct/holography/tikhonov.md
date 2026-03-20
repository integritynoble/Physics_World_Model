# Digital Holographic Microscopy — Tikhonov Regularisation

**CPU**  *Tikhonov A.N., Solution of incorrectly formulated problems and the regularization method, Soviet Mathematics Doklady, 1963*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver
x = run_solver('tikhonov', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
