# Generic Matrix Sensing — ReconNet

**GPU**  *DL for CS reconstruction, 2016*
**Input**: partial matrix (M × N, float32, NaN=missing)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/matrix/public/`

```python
from algorithm_base.matrix.solvers import run_solver
x = run_solver('dl_reconnet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
