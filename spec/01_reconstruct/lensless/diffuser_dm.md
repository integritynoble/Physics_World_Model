# Lensless (Diffuser Camera) Imaging — DiffuserDM

**GPU**  *Diffusion-based generative model for diffuser camera reconstruction, 2023*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('diffuser_dm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
