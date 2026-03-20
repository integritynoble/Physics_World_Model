# Contrast-Enhanced Ultrasound (CEUS) — Med-UNet

**GPU**  *Ronneberger et al., MICCAI 2015*
**Input**: contrast frames (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ceus/public/`

```python
from algorithm_base.ceus.solvers import run_solver
x = run_solver('dl_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
