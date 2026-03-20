# X-ray Computed Tomography (CT) — WGAN-VGG

**GPU**  **PSNR**: ~34.1 dB  *Yang et al., IEEE TMI 2018 — 34.1 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('wgan_vgg', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
