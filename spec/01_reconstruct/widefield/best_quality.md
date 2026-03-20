# Widefield Fluorescence Microscopy — CARE (PnP-PGD DRUNet)

**GPU**  *Weigert et al. 2018, Nature Methods*
**Input**: fluorescence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

```python
from algorithm_base.widefield.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
