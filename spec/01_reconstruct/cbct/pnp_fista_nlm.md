# Cone-Beam Computed Tomography (CBCT) — PnP-FISTA with NLM

**CPU**  *Beck, A. & Teboulle, M. (2009) A fast iterative shrinkage-thresholding algorithm, SIAM J. Imaging Sci. + PnP framework*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('pnp_fista_nlm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
