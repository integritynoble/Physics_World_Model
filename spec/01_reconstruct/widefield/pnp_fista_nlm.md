# Widefield Fluorescence Microscopy — PnP-FISTA (NLM denoiser)

**CPU**  *Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP*
**Input**: fluorescence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

```python
from algorithm_base.widefield.solvers import run_solver
x = run_solver('pnp_fista_nlm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
