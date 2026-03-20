# Lensless (Diffuser Camera) Imaging — PnP-ADMM (NLM)

**CPU**  *Venkatakrishnan S.V. et al., Plug-and-Play Priors for Model Based Reconstruction, IEEE GlobalSIP, 2013*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('pnp_admm_nlm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
