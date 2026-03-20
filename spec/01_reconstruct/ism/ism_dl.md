# Image Scanning Microscopy (ISM) — ISM-Reassignment-Net

**GPU**  *Castello, M. et al. (2019) Image scanning microscopy ISM, Nature Methods 16:175*
**Input**: raw stack (H_scan × W_scan × px × py, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ism/public/`

```python
from algorithm_base.ism.solvers import run_solver
x = run_solver('ism_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
