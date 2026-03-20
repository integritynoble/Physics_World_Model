# Lattice Light-Sheet Microscopy — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lattice_lightsheet/public/`

```python
from algorithm_base.lattice_lightsheet.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
