# X-ray Computed Tomography (CT) — iRadonMAP

**GPU**  **PSNR**: ~36.9 dB  *He et al., MICCAI 2020 — 36.9 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('iradonmap', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
