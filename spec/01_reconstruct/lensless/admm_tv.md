# Lensless (Diffuser Camera) Imaging — ADMM-TV (Lensless)

**CPU**  *Antipa N. et al., DiffuserCam: lensless single-exposure 3D imaging, Optica, 2018*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('admm_tv', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
