# Ultrasound B-mode Imaging — PnP-FISTA (NLM denoiser)

**CPU**  *Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP*
**Input**: RF data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

```python
from algorithm_base.ultrasound.solvers import run_solver
x = run_solver('pnp_fista_nlm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
