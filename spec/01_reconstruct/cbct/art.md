# Cone-Beam Computed Tomography (CBCT) — Algebraic Reconstruction Technique (ART)

**CPU**  *Gordon, R., Bender, R. & Herman, G.T. (1970) Algebraic reconstruction techniques (ART), Journal of Theoretical Biology*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('art', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
