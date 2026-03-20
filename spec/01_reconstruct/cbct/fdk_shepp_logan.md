# Cone-Beam Computed Tomography (CBCT) — FDK Shepp-Logan

**CPU**  *Shepp, L.A. & Logan, B.F. (1974) The Fourier reconstruction of a head section, IEEE Trans. Nuclear Science*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('fdk_shepp_logan', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
