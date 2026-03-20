# Compressed Ultrafast Photography (CUP) — ReconNet

**GPU**  *DL for CS reconstruction, 2016*
**Input**: streak image (H × W_streak, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cup/public/`

```python
from algorithm_base.cup.solvers import run_solver
x = run_solver('dl_reconnet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
