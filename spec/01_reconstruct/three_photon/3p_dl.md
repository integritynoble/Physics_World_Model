# Three-Photon Microscopy — 3P-Net (CARE)

**GPU**  *Weigert, M. et al. (2018) CARE for 3P deep tissue imaging, Nature Methods 15:1090*
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/three_photon/public/`

```python
from algorithm_base.three_photon.solvers import run_solver
x = run_solver('3p_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
