# Portal Imaging (EPID) — MambaRecon

**GPU**  *SSM for inverse problems, 2026*
**Input**: EPID projection (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/portal_imaging/public/`

```python
from algorithm_base.portal_imaging.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
