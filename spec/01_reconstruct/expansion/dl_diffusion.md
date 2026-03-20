# Expansion Microscopy (ExM) — DiffusionMicro

**GPU**  *Diffusion-based microscopy, 2025*
**Input**: confocal + expansion (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/expansion/public/`

```python
from algorithm_base.expansion.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
