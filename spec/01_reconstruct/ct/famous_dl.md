# X-ray Computed Tomography (CT) — RED-CNN

**CPU**  **PSNR**: ~33.2 dB  *Chen et al., IEEE TMI 2017 — 33.2 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
