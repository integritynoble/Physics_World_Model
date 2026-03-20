# Dual-Energy X-ray Absorptiometry (DEXA) — U-Net Recon

**GPU**  *Ronneberger et al., MICCAI 2015*
**Input**: dual-energy projections (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dexa/public/`

```python
from algorithm_base.dexa.solvers import run_solver
x = run_solver('dl_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
