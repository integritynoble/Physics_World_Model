# Cone-Beam Computed Tomography (CBCT) — CBCT-Mamba (RED-DRUNet)

**GPU**  *Wang, Z. et al. (2024) State-space models for efficient CT reconstruction, Medical Image Analysis*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('cbct_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
