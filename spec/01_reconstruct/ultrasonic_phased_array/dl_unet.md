# Ultrasonic Phased Array (TFM/FMC) — Med-UNet

**GPU**  *Ronneberger et al., MICCAI 2015*
**Input**: FMC data (elem × elem × time, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasonic_phased_array/public/`

```python
from algorithm_base.ultrasonic_phased_array.solvers import run_solver
x = run_solver('dl_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
