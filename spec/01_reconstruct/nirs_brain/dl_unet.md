# Functional Near-Infrared Spectroscopy (fNIRS) — Med-UNet

**GPU**  *Ronneberger et al., MICCAI 2015*
**Input**: optical signal (channels × T, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nirs_brain/public/`

```python
from algorithm_base.nirs_brain.solvers import run_solver
x = run_solver('dl_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
