# Susceptibility-Weighted Imaging (SWI) — SwinIR-Med

**GPU**  *Liang et al., ICCV 2021*
**Input**: phase image (H × W × slices, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/swi/public/`

```python
from algorithm_base.swi.solvers import run_solver
x = run_solver('dl_swinir', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
