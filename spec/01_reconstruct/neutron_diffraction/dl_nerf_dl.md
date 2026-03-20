# Neutron Diffraction — NeRF-DL

**GPU**  *Neural rendering, 2020*
**Input**: pattern (2θ × intensity, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_diffraction/public/`

```python
from algorithm_base.neutron_diffraction.solvers import run_solver
x = run_solver('dl_nerf_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
