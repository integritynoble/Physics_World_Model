# Cone-Beam Computed Tomography (CBCT) — FDK + NLM Post-Processing

**CPU**  *Buades, A., Coll, B. & Morel, J.-M. (2005) A non-local algorithm for image denoising, CVPR*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
