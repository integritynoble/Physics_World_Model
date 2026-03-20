# Cryo-EM Single Particle Analysis — CryoSTAR (PnP-DRS DRUNet)

**GPU**  *Guo et al. 2024, Nature Methods*
**Input**: particle images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/public/`

```python
from algorithm_base.cryo_em.solvers import run_solver
x = run_solver('cryostar', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
