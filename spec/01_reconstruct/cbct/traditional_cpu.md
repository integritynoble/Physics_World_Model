# Cone-Beam Computed Tomography (CBCT) — FDK Ram-Lak

**CPU**  *Feldkamp, L.A., Davis, L.C. & Kress, J.W. (1984) Practical cone-beam algorithm, JOSA A*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
