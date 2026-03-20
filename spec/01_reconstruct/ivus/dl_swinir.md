# Intravascular Ultrasound (IVUS) — SwinIR-Med

**GPU**  *Liang et al., ICCV 2021*
**Input**: RF pullback (frames × elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ivus/public/`

```python
from algorithm_base.ivus.solvers import run_solver
x = run_solver('dl_swinir', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
