# Ground-Penetrating Radar (GPR) — RS-Mamba

**GPU**  *SSM for remote sensing, 2026*
**Input**: B-scan (traces × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/gpr/public/`

```python
from algorithm_base.gpr.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
