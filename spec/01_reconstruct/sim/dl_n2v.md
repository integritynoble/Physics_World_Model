# Structured Illumination Microscopy (SIM) — Noise2Void

**GPU**  *Krull et al., CVPR 2019*
**Input**: raw frames (9 × H × W: 3 angles × 3 phases)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sim/public/`

```python
from algorithm_base.sim.solvers import run_solver
x = run_solver('dl_n2v', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
