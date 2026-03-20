# Compressed Ultrafast Photography (CUP) — CS-Diffusion

**GPU**  *Diffusion for CS, 2025*
**Input**: streak image (H × W_streak, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cup/public/`

```python
from algorithm_base.cup.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
