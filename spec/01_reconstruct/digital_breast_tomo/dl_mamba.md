# Digital Breast Tomosynthesis (DBT) — MambaRecon

**GPU**  *SSM for inverse problems, 2026*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/digital_breast_tomo/public/`

```python
from algorithm_base.digital_breast_tomo.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
