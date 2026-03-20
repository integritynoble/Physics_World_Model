# Cryo-EM Single Particle Analysis — PnP-ADMM (NLM denoiser)

**CPU**  *Venkatakrishnan et al. 2013, GlobalSIP*
**Input**: particle images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/public/`

```python
from algorithm_base.cryo_em.solvers import run_solver
x = run_solver('pnp_admm_nlm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
