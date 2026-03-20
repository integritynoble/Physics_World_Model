# Cryo-EM Single Particle Analysis — SIRT (Simultaneous Iterative)

**CPU**  *Gilbert 1972, J. Theor. Biol.*
**Input**: particle images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/public/`

```python
from algorithm_base.cryo_em.solvers import run_solver
x = run_solver('sirt_3d', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
