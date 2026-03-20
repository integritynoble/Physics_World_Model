# STED Microscopy — RCAN-STED

**GPU**  *Chen, J. et al. (2021) Three-dimensional residual channel attention for STED, Nature Methods 18:678*
**Input**: STED + confocal (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sted/public/`

```python
from algorithm_base.sted.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
