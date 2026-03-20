# X-ray Computed Tomography (CT) — FBPConvNet

**GPU**  **PSNR**: ~38.5 dB  *Jin et al., TIP 2017 — 38.5 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('fbpconvnet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
