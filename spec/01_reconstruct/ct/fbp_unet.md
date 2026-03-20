# X-ray Computed Tomography (CT) — FBP + U-Net

**GPU**  **PSNR**: ~35.8 dB  *Ronneberger et al. 2015 / Jin et al. 2017 — 35.8 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('fbp_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
