# Contrast-Enhanced Ultrasound (CEUS) — SwinIR-Med

**GPU**  *Liang et al., ICCV 2021*
**Input**: contrast frames (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ceus/public/`

```python
from algorithm_base.ceus.solvers import run_solver
x = run_solver('dl_swinir', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
