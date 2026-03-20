# Cone-Beam Computed Tomography (CBCT) — FDK-DL (DL-PGD)

**GPU**  *Chen, H. et al. (2017) Low-dose CT with a residual encoder-decoder CNN, IEEE TMI*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
