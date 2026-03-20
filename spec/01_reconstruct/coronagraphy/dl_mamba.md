# Stellar Coronagraphy — DL-Mamba

**GPU**  *SSM reconstruction, 2026*
**Input**: image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/coronagraphy/public/`

```python
from algorithm_base.coronagraphy.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
