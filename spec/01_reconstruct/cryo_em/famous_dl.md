# Cryo-EM Single Particle Analysis — CryoDRGN (PnP-PGD DRUNet)

**GPU**  *Zhong et al. 2021, Nature Methods*
**Input**: particle images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/public/`

```python
from algorithm_base.cryo_em.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
