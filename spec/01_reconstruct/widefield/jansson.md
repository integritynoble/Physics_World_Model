# Widefield Fluorescence Microscopy — Jansson-van Cittert Iteration

**CPU**  *van Cittert 1931, Zeitschrift f. Physik; Jansson 1970*
**Input**: fluorescence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

```python
from algorithm_base.widefield.solvers import run_solver
x = run_solver('jansson', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
