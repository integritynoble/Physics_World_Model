# STED Microscopy — DiffusionMicro

**GPU**  *Diffusion-based microscopy, 2025*
**Input**: STED + confocal (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sted/public/`

```python
from algorithm_base.sted.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
