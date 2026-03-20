# Optical Diffraction Tomography (ODT) — Adjoint [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: holograms (angles × H × W, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/odt/public/`

```python
from algorithm_base.odt.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
