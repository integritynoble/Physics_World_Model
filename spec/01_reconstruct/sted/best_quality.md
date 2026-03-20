# STED Microscopy — STED-Net (CARE)

**GPU**  *Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090*
**Input**: STED + confocal (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sted/public/`

```python
from algorithm_base.sted.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
