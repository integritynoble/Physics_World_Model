# Dark-Field Microscopy — DF-UNet

**GPU**  *Wolfer, T. et al. (2021) DL for dark-field X-ray CT, Sci. Rep. 11:5005*
**Input**: grating images (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dark_field/public/`

```python
from algorithm_base.dark_field.solvers import run_solver
x = run_solver('df_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
