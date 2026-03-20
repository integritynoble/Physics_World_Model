# Fluoroscopy — Med-UNet

**GPU**  *Ronneberger et al., MICCAI 2015*
**Input**: X-ray frames (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fluoroscopy/public/`

```python
from algorithm_base.fluoroscopy.solvers import run_solver
x = run_solver('dl_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
