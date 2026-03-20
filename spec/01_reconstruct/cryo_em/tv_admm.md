# Cryo-EM Single Particle Analysis — Total Variation ADMM

**CPU**  *Boyd et al. 2011, ADMM; Rudin-Osher-Fatemi 1992 TV*
**Input**: particle images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/public/`

```python
from algorithm_base.cryo_em.solvers import run_solver
x = run_solver('tv_admm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
