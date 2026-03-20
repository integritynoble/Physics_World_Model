# Small-Angle X-ray Scattering (SAXS) — prDeep

**GPU**  *Deep phase retrieval, 2020*
**Input**: scattering pattern (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/saxs/public/`

```python
from algorithm_base.saxs.solvers import run_solver
x = run_solver('dl_prdeep', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
