# Cryo-EM Single Particle Analysis — CryoDRGN2 (PnP-HQS DRUNet)

**GPU**  *Zhong et al. 2021, ICLR*
**Input**: particle images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/public/`

```python
from algorithm_base.cryo_em.solvers import run_solver
x = run_solver('cryodrgn2', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
