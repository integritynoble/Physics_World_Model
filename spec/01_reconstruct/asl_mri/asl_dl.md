# Arterial Spin Labeling (ASL) MRI — ASL-Net [proxy]

**CPU**
**Input**: label-control pairs (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/asl_mri/public/`

```python
from algorithm_base.asl_mri.solvers import run_solver
x = run_solver('asl_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
