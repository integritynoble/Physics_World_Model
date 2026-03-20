# Cryo-EM Single Particle Analysis — RELION (PnP-PGD DRUNet)

**GPU**  *Scheres 2012, JMB; Zivanov et al. 2018, eLife*
**Input**: particle images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/public/`

```python
from algorithm_base.cryo_em.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
