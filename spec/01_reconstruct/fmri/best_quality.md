# Functional MRI (BOLD fMRI) — fMRI-Transformer [proxy]

**CPU**
**Input**: BOLD volumes (T × H × W × D, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fmri/public/`

```python
from algorithm_base.fmri.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
