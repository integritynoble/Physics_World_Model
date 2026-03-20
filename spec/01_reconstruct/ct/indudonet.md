# X-ray Computed Tomography (CT) — InDuDoNet

**GPU**  **PSNR**: ~43.5 dB  *Song et al., MICCAI 2021 — 43.5 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('indudonet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
