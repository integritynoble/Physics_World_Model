# Lensless (Diffuser Camera) Imaging — PnP-HQS (NLM)

**CPU**  *Zhang K. et al., Learning Deep CNN Denoiser Prior for Image Restoration, CVPR, 2017*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('pnp_hqs_nlm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
