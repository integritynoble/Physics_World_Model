# X-ray Computed Tomography (CT) — DuDoTrans

**GPU**  **PSNR**: ~42.1 dB  *Wang et al., MICCAI 2022 — 42.1 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('dudotrans', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
