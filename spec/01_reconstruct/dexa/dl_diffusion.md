# Dual-Energy X-ray Absorptiometry (DEXA) — DiffusionRecon

**GPU**  *Song et al., 2024*
**Input**: dual-energy projections (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dexa/public/`

```python
from algorithm_base.dexa.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
