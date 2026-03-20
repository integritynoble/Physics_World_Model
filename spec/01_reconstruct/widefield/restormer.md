# Widefield Fluorescence Microscopy — Restormer (PnP-HQS DRUNet)

**GPU**  *Zamir et al. 2022, CVPR*
**Input**: fluorescence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

```python
from algorithm_base.widefield.solvers import run_solver
x = run_solver('restormer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
