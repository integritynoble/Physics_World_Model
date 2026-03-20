# Intravascular Ultrasound (IVUS) — IVUS-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: RF pullback (frames × elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ivus/public/`

```python
from algorithm_base.ivus.solvers import run_solver
x = run_solver('ivus_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
