# Fluorescence Lifetime Imaging (FLIM) — Noise2Void

**GPU**  *Krull et al., CVPR 2019*
**Input**: photon arrivals (H × W × T, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/flim/public/`

```python
from algorithm_base.flim.solvers import run_solver
x = run_solver('dl_n2v', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
