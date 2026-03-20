# X-ray Computed Tomography (CT) — DuDoNet

**GPU**  **PSNR**: ~40.2 dB  *Lin et al., CVPR 2019 — 40.2 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('dudonet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
