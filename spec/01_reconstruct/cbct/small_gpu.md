# Cone-Beam Computed Tomography (CBCT) — CBCT-UNet (DnCNN)

**GPU**  *Jin, K.H. et al. (2017) Deep convolutional neural network for inverse problems in imaging, IEEE TIP*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('small_gpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
