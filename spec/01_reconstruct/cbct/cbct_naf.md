# Cone-Beam Computed Tomography (CBCT) — CBCT Neural Attenuation Fields (DL-DRS)

**GPU**  *Zha, R. et al. (2024) NAF: Neural Attenuation Fields for sparse-view CBCT reconstruction, IEEE TMI*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('cbct_naf', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
