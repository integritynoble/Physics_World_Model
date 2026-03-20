# Cone-Beam Computed Tomography (CBCT) — ML-EM

**CPU**  *Shepp, L.A. & Vardi, Y. (1982) Maximum likelihood reconstruction for emission tomography, IEEE TMI*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('mlem', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
