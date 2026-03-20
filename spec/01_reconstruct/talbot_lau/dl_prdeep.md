# Talbot-Lau X-ray Grating Interferometry — prDeep

**GPU**  *Deep phase retrieval, 2020*
**Input**: stepping images (N_steps × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/talbot_lau/public/`

```python
from algorithm_base.talbot_lau.solvers import run_solver
x = run_solver('dl_prdeep', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
