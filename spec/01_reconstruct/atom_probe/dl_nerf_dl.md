# Atom Probe Tomography (APT) — NeRF-DL

**GPU**  *Neural rendering, 2020*
**Input**: hit positions (N × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/atom_probe/public/`

```python
from algorithm_base.atom_probe.solvers import run_solver
x = run_solver('dl_nerf_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
