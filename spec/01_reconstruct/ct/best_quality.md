# X-ray Computed Tomography (CT) — FBP + NLM

**CPU**  **PSNR**: ~28.5 dB  *Buades, Coll & Morel, CVPR 2005 — 28.5 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
