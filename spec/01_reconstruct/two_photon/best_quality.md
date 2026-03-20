# Two-Photon / Multiphoton Microscopy — 2P-Net (CARE)

**GPU**  *Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090*
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/two_photon/public/`

```python
from algorithm_base.two_photon.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
