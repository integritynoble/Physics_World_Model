# Two-Photon / Multiphoton Microscopy — Restormer

**GPU**  *Zamir et al., CVPR 2022*
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/two_photon/public/`

```python
from algorithm_base.two_photon.solvers import run_solver
x = run_solver('dl_restormer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
