# Susceptibility-Weighted Imaging (SWI) — Med-UNet

**GPU**  *Ronneberger et al., MICCAI 2015*
**Input**: phase image (H × W × slices, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/swi/public/`

```python
from algorithm_base.swi.solvers import run_solver
x = run_solver('dl_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
