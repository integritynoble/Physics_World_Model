# Lensless (Diffuser Camera) Imaging — L3Fnet

**GPU**  *Tan G. et al., L3Fnet: Lensless Light-Field Reconstruction Network, IEEE TMM, 2023*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('l3fnet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
