# Fundus Camera — DR-Grade-Net

**GPU**  *Gulshan, V. et al. (2016) DL for DR detection in retinal fundus, JAMA 316(22)*
**Input**: photograph (H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fundus/public/`

```python
from algorithm_base.fundus.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
