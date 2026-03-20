# Widefield Fluorescence Microscopy — Richardson-Lucy with TV Regularisation

**CPU**  *Dey et al. 2006, Microscopy Res. Tech.*
**Input**: fluorescence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

```python
from algorithm_base.widefield.solvers import run_solver
x = run_solver('rl_tv', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
