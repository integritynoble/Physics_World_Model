# Arterial Spin Labeling (ASL) MRI — FBP [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: label-control pairs (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/asl_mri/public/`

```python
from algorithm_base.asl_mri.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
