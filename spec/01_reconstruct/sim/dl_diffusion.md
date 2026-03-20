# Structured Illumination Microscopy (SIM) — DiffusionMicro

**GPU**  *Diffusion-based microscopy, 2025*
**Input**: raw frames (9 × H × W: 3 angles × 3 phases)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sim/public/`

```python
from algorithm_base.sim.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
